from pathlib import Path
import json
import math
import os

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from he_aware_training import PROJECT_ROOT
from he_aware_training.utils.checkpoint import load_init_checkpoint

@torch.no_grad()
def _causal_attn(block: GPT2Block, normed: torch.Tensor):
    B, T, C = normed.shape
    H = block.attn.num_heads
    d = C // H

    qkv = block.attn.c_attn(normed)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, H, d).transpose(1, 2)
    k = k.view(B, T, H, d).transpose(1, 2)
    v = v.view(B, T, H, d).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device)).view(1, 1, T, T)
    pre_softmax  = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    post_softmax = torch.softmax(pre_softmax, dim=-1)

    ctx = post_softmax @ v                                 # (B, H, T, d)
    ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
    attn_out = block.attn.c_proj(ctx)
    return attn_out, pre_softmax, post_softmax


@torch.no_grad()
def block_forward(block: GPT2Block, hs: torch.Tensor) -> torch.Tensor:
    residual = hs
    normed = block.ln_1(hs)
    attn_out, _, _ = _causal_attn(block, normed)
    r1 = attn_out + residual

    normed2 = block.ln_2(r1)
    pre_gelu = block.mlp.c_fc(normed2)
    post_gelu = block.mlp.act(pre_gelu)
    down = block.mlp.c_proj(post_gelu)

    return r1 + down


@torch.no_grad()
def block_forward_with_taps(block: GPT2Block, hs: torch.Tensor):
    residual = hs
    normed = block.ln_1(hs)

    attn_out, pre_softmax, post_softmax = _causal_attn(block, normed)
    r1 = attn_out + residual

    normed2 = block.ln_2(r1)
    pre_gelu = block.mlp.c_fc(normed2)
    post_gelu = block.mlp.act(pre_gelu)
    down = block.mlp.c_proj(post_gelu)
    out = r1 + down

    return {
        "inp":          hs[0],
        "ln_1_out":     normed[0],
        "pre_softmax":  pre_softmax[0],
        "post_softmax": post_softmax[0],
        "attn_out":     attn_out[0],
        "r1":           r1[0],
        "ln_2_out":     normed2[0],
        "pre_gelu":     pre_gelu[0],
        "post_gelu":    post_gelu[0],
        "mlp_out":      down[0],
        "res":          out[0],
    }, out


def capture_first_block_input(model):
    captured = {}
    blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, GPT2Block)]
    if not blocks:
        raise RuntimeError("No GPT2Block modules found.")
    first_name, first_block = blocks[0]

    def hook(mod, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if not torch.is_tensor(hs):
            return
        captured["hs"] = hs.detach()
        raise StopIteration

    handle = first_block.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, handle, [m for _, m in blocks], first_name


@hydra.main(version_base=None,
            config_path=f"{PROJECT_ROOT}/configs",
            config_name="calibrate")
def main(cfg: DictConfig):
    torch.manual_seed(int(os.environ.get("GATHER_SEED", 42)))  

    seq_lengths = [1, 2, 4, 8, 12, 16, 32, 64, 128, 512]
    if os.environ.get("GATHER_SEQ_LENGTHS"):  
        seq_lengths = [int(t) for t in os.environ["GATHER_SEQ_LENGTHS"].split(",")]
    out_dir = Path(os.environ.get("ALL_BLOCKS_IO_DIR", f"{PROJECT_ROOT}/data/all_blocks_io"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = instantiate(cfg.model.instance)
    if model.config.vocab_size != cfg.model.vocab_size:
        model.resize_token_embeddings(cfg.model.vocab_size)
    model.to(device=cfg.device, dtype=torch.float32).eval()
    if OmegaConf.select(cfg, "trainer.init_from"):     
        load_init_checkpoint(model, cfg, cfg.device,   
                             transpose_keys=(".attn.c_proj.weight",))  
    get_batch, _ = instantiate(cfg.dataset.get_batch)

    captured, handle, blocks, first_name = capture_first_block_input(model)
    X, _ = get_batch("train")
    try:
        model(X)
    except StopIteration:
        pass
    finally:
        handle.remove()

    if "hs" not in captured:
        raise RuntimeError("Forward pre-hook did not fire on the first GPT2Block.")
    hs0 = captured["hs"][:1]                       # (1, T_full, C)
    C = hs0.shape[-1]
    T_full = hs0.shape[1]
    print(f"[capture] embed input={tuple(hs0.shape)} on {first_name}; "
          f"layers={len(blocks)}, heads={blocks[0].attn.num_heads}, "
          f"head_dim={C // blocks[0].attn.num_heads}")

    ln_f = next((m for n, m in model.named_modules() if n.endswith("ln_f")), None)
    if ln_f is None:
        raise RuntimeError("No final layer norm (…ln_f) found on the model.")

    for T in seq_lengths:
        if T > T_full:
            print(f"[skip] T={T} > block_size={T_full}")
            continue
        hs_T = hs0[:, :T, :]
        cur_in = hs_T
        for L, block in enumerate(blocks):
            out_path = out_dir / f"all_blocks_L{L:02d}_T{T}.json"
            if L == 0:
                taps, cur_out = block_forward_with_taps(block, cur_in)
                payload = {k: v.float().cpu().tolist() for k, v in taps.items()}
                inp_t = taps["inp"].float()
                res_t = taps["res"].float()
                with open(out_path, "w") as f:
                    json.dump(payload, f)
                print(f"[save] L={L:>2}  T={T:>5}  C={C}  taps={list(taps)}  "
                      f"inp_range=[{float(inp_t.min()):+.3f}, {float(inp_t.max()):+.3f}]  "
                      f"out_range=[{float(res_t.min()):+.3f}, {float(res_t.max()):+.3f}]  → {out_path}")
            else:
                taps, cur_out = block_forward_with_taps(block, cur_in)
                inp = taps["inp"].float().cpu()
                res = taps["res"].float().cpu()
                payload = {
                    "inp":      inp.tolist(),
                    "ln_1_out": taps["ln_1_out"].float().cpu().tolist(),
                    "attn_out": taps["attn_out"].float().cpu().tolist(),
                    "r1":       taps["r1"].float().cpu().tolist(),
                    "ln_2_out": taps["ln_2_out"].float().cpu().tolist(),
                    "pre_gelu": taps["pre_gelu"].float().cpu().tolist(),
                    "mlp_out":  taps["mlp_out"].float().cpu().tolist(),
                    "res":      res.tolist(),
                }
                if T <= 128:
                    payload["pre_softmax"]  = taps["pre_softmax"].float().cpu().tolist()
                    payload["post_softmax"] = taps["post_softmax"].float().cpu().tolist()
                with open(out_path, "w") as f:
                    json.dump(payload, f)
                print(f"[save] L={L:>2}  T={T:>5}  C={C}  +attn/r1/mlp"
                      f"{'+softmax' if T <= 128 else ''} taps  "
                      f"inp_range=[{float(inp.min()):+.3f}, {float(inp.max()):+.3f}]  "
                      f"out_range=[{float(res.min()):+.3f}, {float(res.max()):+.3f}]  → {out_path}")
            cur_in = cur_out

        with torch.no_grad():
            final = ln_f(cur_out)
        lnf_path = out_dir / f"all_blocks_ln_f_T{T}.json"
        pre = cur_out[0].float().cpu()
        post = final[0].float().cpu()
        with open(lnf_path, "w") as f:
            json.dump({"inp": pre.tolist(), "res": post.tolist()}, f)
        print(f"[save] ln_f  T={T:>5}  C={C}  "
              f"inp_range=[{float(pre.min()):+.3f}, {float(pre.max()):+.3f}]  "
              f"out_range=[{float(post.min()):+.3f}, {float(post.max()):+.3f}]  → {lnf_path}")

        with torch.no_grad():
            logits = model.lm_head(final)            # (1, T, vocab)
        last = logits[0, -1].float().cpu()           # (vocab,)
        k = min(10, last.numel())
        topk = torch.topk(last, k)
        lm_path = out_dir / f"all_blocks_lm_head_T{T}.json"
        with open(lm_path, "w") as f:
            json.dump({
                "logits":   last.tolist(),
                "argmax":   int(last.argmax().item()),
                "topk_idx": topk.indices.tolist(),
                "topk_val": topk.values.tolist(),
            }, f)
        print(f"[save] lm_head  T={T:>5}  vocab={last.numel()}  "
              f"argmax={int(last.argmax().item())}  "
              f"top1_logit={float(last.max()):+.3f}  → {lm_path}")

    steps_T = min(T_full, int(cfg.get("steps_T", 16)))
    cur = hs0[:, :steps_T, :]
    for block in blocks:
        cur = block_forward(block, cur)
    with torch.no_grad():
        final = ln_f(cur)
        logits = model.lm_head(final)[0].float().cpu()   # (steps_T, vocab)
    per_step = []
    for p in range(steps_T):
        row = logits[p]
        topk = torch.topk(row, min(10, row.numel()))
        per_step.append({
            "logits":   row.tolist(),
            "argmax":   int(row.argmax().item()),
            "topk_idx": topk.indices.tolist(),
            "topk_val": topk.values.tolist(),
        })
    steps_path = out_dir / f"all_blocks_lm_head_steps_T{steps_T}.json"
    with open(steps_path, "w") as f:
        json.dump({"T": steps_T, "vocab": int(logits.shape[1]), "steps": per_step}, f)
    print(f"[save] lm_head_steps  T={steps_T:>5}  vocab={int(logits.shape[1])}  "
          f"argmax_per_pos={[s['argmax'] for s in per_step]}  → {steps_path}")


if __name__ == "__main__":
    main()
