import json
from pathlib import Path

import torch
import wandb
import lm_eval
from lm_eval.models.huggingface import HFLM
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from he_aware_training import PROJECT_ROOT
from he_aware_training.utils.surgery import perform_he_surgery
from he_aware_training.utils.checkpoint import load_checkpoint

TASK_PRIMARY_METRIC = {
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "mnli": "acc,none",
    "sst2": "acc,none",
    "anli_r1": "acc,none",
    "anli_r2": "acc,none",
    "anli_r3": "acc,none",
    "wic": "acc,none",
    "swag": "acc_norm,none",
    "cola": "acc,none",
    "mrpc": "acc,none",
    "qqp": "acc,none",
    "mnli_mismatch": "acc,none",
    "qnli": "acc,none",
    "rte": "acc,none",
    "wnli": "acc,none",
}

TASK_SECONDARY_METRIC = {
    "cola": "mcc,none",
}


def load_hybrid_model(cfg, device):
    """Build the hybrid HE-friendly GPT-2: instantiate base model → perform_he_surgery
    (THOR softmax + softsign GELU + LN approximation) → optionally load a trained
    checkpoint."""
    model = instantiate(cfg.model.instance)
    if model.config.vocab_size != cfg.model.vocab_size:
        model.resize_token_embeddings(cfg.model.vocab_size)

    model = perform_he_surgery(model, cfg, verbose=True)

    if cfg.eval.backbone_ckpt is not None:
        print(f"  Loading hybrid checkpoint from: {cfg.eval.backbone_ckpt}")
        # strict=False: approximation params (inv_sqrt_approx/ponder) come from calib_path, not the ckpt
        load_checkpoint(cfg.eval.backbone_ckpt, model, optimizer=None, device=device, strict=False)

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if __import__("os").environ.get("FHE_NAN_TAP"):
        _st = {"fired": False}
        def _mk(nm):
            def _h(m, inp, out):
                if _st["fired"]:
                    return
                o = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(o) and not torch.isfinite(o).all():
                    ins = inp if isinstance(inp, tuple) else (inp,)
                    fin = all((not torch.is_tensor(x)) or torch.isfinite(x).all() for x in ins)
                    print(f"[NAN_TAP] {'FIRST finite-in->NaN-out' if fin else 'propagated'}: {nm} ({type(m).__name__})", flush=True)
                    if fin:
                        _st["fired"] = True
            return _h
        for nm, mod in model.named_modules():
            if not list(mod.children()):
                mod.register_forward_hook(_mk(nm))
        print("[NAN_TAP] armed (leaf-module NaN localizer)", flush=True)

    if __import__("os").environ.get("FHE_D_TAP"):
        _acc = {}
        _smods = {nm: m for nm, m in model.named_modules()
                  if type(m).__name__ == "THORLearnableSoftmax"}
        def _smk():
            def _h(m, args, kwargs):
                x = kwargs.get("x", args[0] if args else None)
                cm = kwargs.get("causal_mask", args[1] if len(args) > 1 else None)
                if not torch.is_tensor(x):
                    return
                xf = x.detach().float()
                keep = (~cm).float() if torch.is_tensor(cm) else torch.ones_like(xf)
                ke = keep.sum(-1, keepdim=True).clamp(min=1.0)
                xv = xf.masked_fill(keep == 0, float("-inf"))
                m._tap = ((xf.exp() * keep).sum(-1, keepdim=True), ke, float(xv.max()))
            return _h
        for _m in _smods.values():
            _m.register_forward_pre_hook(_smk(), with_kwargs=True)
        def _qs(t):
            f = t.flatten().float()
            if f.numel() > 200000:
                f = f[torch.randperm(f.numel())[:200000]]
            return [round(float(torch.quantile(f, q)), 3) for q in (0.5, 0.99, 0.999, 1.0)]
        def _pmk(nm):
            pnm = nm.rsplit(".", 1)[0]
            def _h(m, args, kwargs):
                a = _acc.setdefault(pnm, {"ut": [], "ua": [], "xmax": 0.0, "n": 0, "done": False})
                if a["done"]:
                    return
                den = args[1] if len(args) > 1 else None
                tup = getattr(_smods.get(pnm), "_tap", None)
                if not (torch.is_tensor(den) and tup):
                    return
                tsum, ke, xmax = tup
                a["ut"].append((tsum / ke).flatten().cpu())
                a["ua"].append((den.float() / ke).flatten().cpu())
                a["xmax"] = max(a["xmax"], xmax)
                a["n"] += 1
                if a["n"] >= 8:
                    ut = torch.cat(a["ut"]); ua = torch.cat(a["ua"])
                    print(f"[D_DIST] {pnm}: att_x.max={a['xmax']:.2f} | u_true q[.5,.99,.999,max]={_qs(ut)} | u_approx={_qs(ua)}  (basin~9.7)", flush=True)
                    a["done"] = True; a["ut"] = []; a["ua"] = []
            return _h
        for nm, mod in model.named_modules():
            if nm.endswith("ponder_goldschmidt_init"):
                mod.register_forward_pre_hook(_pmk(nm), with_kwargs=True)
        print("[D_DIST] armed (denominator distribution tap)", flush=True)
    return model


def make_lm(model_or_name, tokenizer=None, device="cuda", batch_size=8):
    """Wrap a model (or HF hub name) in an HFLM for lm-eval."""
    kwargs = dict(backend="causal", device=device, batch_size=batch_size)
    if isinstance(model_or_name, str):
        return HFLM(pretrained=model_or_name, **kwargs)
    return HFLM(pretrained=model_or_name, tokenizer=tokenizer, **kwargs)


def extract_scores(lm_results, tasks):
    """Primary per-task metric (uniform accuracy for GLUE) from lm_eval output."""
    results = lm_results.get("results", {})
    scores = {}
    for task in tasks:
        task_res = results.get(task, {})
        preferred = TASK_PRIMARY_METRIC.get(task, "acc,none")
        for metric in (preferred, "acc_norm,none", "acc,none"):
            if metric in task_res:
                scores[task] = round(task_res[metric], 4)
                break
    return scores


def extract_secondary(lm_results, tasks):
    """Secondary metrics (e.g. CoLA Matthews corr) kept beside accuracy, not averaged."""
    results = lm_results.get("results", {})
    extra = {}
    for task in tasks:
        metric = TASK_SECONDARY_METRIC.get(task)
        if metric and metric in results.get(task, {}):
            extra[f"{task} ({metric.split(',')[0]})"] = round(results[task][metric], 4)
    return extra


def dump_results(lm_results, path):
    """Persist the full per-task metric dict so no metric is lost to a re-run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(lm_results.get("results", {}), f, indent=2, default=str)
    print(f"[save] full results → {path}")


def print_table(all_scores, tasks, all_extra=None):
    model_names = list(all_scores.keys())
    col_w = 14
    header = f"{'task':22s}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    def cell(v):
        return f"{v:>{col_w}.4f}" if isinstance(v, (int, float)) and v == v else f"{'N/A':>{col_w}}"

    for task in tasks:
        row = f"  {task:20s}" + "".join(cell(all_scores[m].get(task)) for m in model_names)
        print(row)
    all_extra = all_extra or {}
    extra_labels = sorted({k for e in all_extra.values() for k in e})
    if extra_labels:
        print("─" * len(header))
        for label in extra_labels:
            row = f"  {label:20s}" + "".join(cell(all_extra.get(m, {}).get(label)) for m in model_names)
            print(row)
    print("─" * len(header))
    for m in model_names:
        vals = [all_scores[m][t] for t in tasks if isinstance(all_scores[m].get(t), float)]
        avg = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {'avg acc (' + m + ')':30s}{avg:>{col_w}.4f}")


@hydra.main(
    version_base=None,
    config_path=f"{PROJECT_ROOT}/configs",
    config_name="lm_eval",
)
def main(cfg: DictConfig):
    device = cfg.device
    tasks = list(cfg.eval.tasks)
    batch_size = cfg.eval.batch_size
    num_fewshot = cfg.eval.num_fewshot

    wandb.init(
        entity=cfg.core.entity,
        project=cfg.core.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.core.tags),
    )

    print(f"\n{'='*60}")
    print(f"LM BENCHMARKS  num_fewshot={num_fewshot}  tasks={tasks}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = tokenizer.eos_token

    all_scores = {}
    all_extra = {}
    out_dir = Path(cfg.eval.output_dir)

    baseline_name = cfg.model.name
    baseline_key = baseline_name.replace("/", "_").replace("-", "_")
    if cfg.eval.run_baseline:
        print(f"─── Evaluating: {baseline_name} (pretrained baseline) ───")
        lm = make_lm(baseline_name, device=device, batch_size=batch_size)
        raw = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=num_fewshot, log_samples=False,
        )
        all_scores[baseline_key] = extract_scores(raw, tasks)
        all_extra[baseline_key] = extract_secondary(raw, tasks)
        dump_results(raw, out_dir / f"{baseline_key}_results.json")
        del lm
        torch.cuda.empty_cache()

    if cfg.eval.run_hybrid:
        print(f"\n─── Evaluating: hybrid (THOR softmax + softsign GELU + LN approx) ───")
        hybrid_model = load_hybrid_model(cfg, device)
        lm = make_lm(hybrid_model, tokenizer=tokenizer, device=device, batch_size=batch_size)
        raw = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=num_fewshot, log_samples=False,
        )
        all_scores["hybrid"] = extract_scores(raw, tasks)
        all_extra["hybrid"] = extract_secondary(raw, tasks)
        ckpt = cfg.eval.backbone_ckpt
        if ckpt and str(ckpt) != "null":
            p = Path(str(ckpt))
            hybrid_fname = f"hybrid_{p.parent.name}_{p.stem}_results.json"
        else:
            hybrid_fname = "hybrid_results.json"
        dump_results(raw, out_dir / hybrid_fname)
        del lm, hybrid_model
        torch.cuda.empty_cache()

    print_table(all_scores, tasks, all_extra)

    table = wandb.Table(columns=["task"] + list(all_scores.keys()))
    for task in tasks:
        row = [task] + [all_scores[m].get(task, float("nan")) for m in all_scores]
        table.add_data(*row)
    wandb.log({"comparison_table": table})

    for model_name, scores in all_scores.items():
        vals = [scores[t] for t in tasks if isinstance(scores.get(t), float)]
        wandb.log(
            {
                **{f"{model_name}/{task}": acc for task, acc in scores.items()},
                **{f"{model_name}/{lbl}": v for lbl, v in all_extra.get(model_name, {}).items()},
                f"{model_name}/avg_acc": sum(vals) / len(vals) if vals else 0.0,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
