import os
import time

import numpy as np
import torch


def build_run_tags(cfg) -> list:
    tags = []

    tags.append(f"prior:{cfg.prior.name}")
    tags.append(f"inference:{cfg.prior.inference_mode}")
    tags.append(f"decr:{cfg.prior.decr_trigger}")

    approx_name = cfg.model.approximation.get("name")
    if approx_name:
        tags.append(f"approx:{approx_name}")

    if approx_name == "enc_llm":
        tags.append(f"softmax-init:{cfg.model.approximation.enc_llm.g_init_mode}")

    surgery = cfg.model.surgery
    short = {"norm": "norm", "activation": "act", "attention": "attn"}
    active = [short[k] for k in ("norm", "activation", "attention") if surgery.get(k)]
    tags.append("surgery:" + ("+".join(active) if active else "none"))

    tags.append(f"model:{cfg.model.name.split('/')[-1]}")

    return tags


def freeze_model_components(model, cfg):
    import re

    print("[Freezing] Applying freeze policy...")
    freeze = cfg.model.freeze

    # Determine which encoder layer indices to freeze
    encoder_layers_to_train = getattr(freeze, "encoder_layers_to_train", None)
    if encoder_layers_to_train is not None:
        all_indices = sorted(
            {
                int(m.group(1))
                for name, _ in model.named_parameters()
                for m in [re.search(r"encoder\.layer\.(\d+)", name)]
                if m
            }
        )
        freeze_up_to = max(all_indices) - encoder_layers_to_train  # inclusive

    for name, param in model.named_parameters():
        param.requires_grad = True
        if freeze.embeddings:
            # GPT2: wte, wpe, shared | ViT: vit.embeddings
            if (
                "wte" in name
                or "wpe" in name
                or "shared" in name
                or "embeddings" in name
            ):
                param.requires_grad = False
                print(f"[Freezing] {name}")
        if getattr(freeze, "lm_head", False):
            if "lm_head" in name:
                param.requires_grad = False
                print(f"[Freezing] {name}")
        if getattr(freeze, "classifier", False):
            if "classifier" in name:
                param.requires_grad = False
                print(f"[Freezing] {name}")
        if encoder_layers_to_train is not None:
            m = re.search(r"encoder\.layer\.(\d+)", name)
            if m and int(m.group(1)) <= freeze_up_to:
                param.requires_grad = False
                print(f"[Freezing] {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Freezing] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)"
    )
    return model


@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_iters, device, teacher=None):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        t_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            if X is None:
                losses[k] = 0.0
                continue

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(X, labels=Y)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[1]
                if teacher is not None and split == "val":
                    t_out = teacher(X, labels=Y)
                    t_losses[k] = (t_out.loss if hasattr(t_out, "loss") else t_out[1]).item()
            losses[k] = loss.item()
        finite = torch.isfinite(losses)
        out[split] = losses[finite].mean() if finite.any() else losses.mean()
        out[f"{split}_nonfinite"] = int((~finite).sum().item())
        if teacher is not None and split == "val":
            out["val_teacher"] = t_losses.mean()
    model.train()
    return out

def compose_get_batch(model_name, model_dtype, data_path, dataset_name, device, block_size, micro_batch):
    data_dir = os.path.join(data_path, dataset_name)
    dtype_in = np.uint32 if model_dtype == "uint32" else np.uint16

    model_name_clean = model_name.split("/")[-1]
    potential_train = f"{model_name_clean}_train.bin"
    if os.path.exists(os.path.join(data_dir, potential_train)):
        train_file = potential_train
        val_file = f"{model_name_clean}_val.bin"
    else:
        raise FileNotFoundError(f"Could not find data in {data_dir}")

    def get_batch(split):
        filename = train_file if split == "train" else val_file
        path = os.path.join(data_dir, filename)
        for attempt in range(5):  
            try:
                data = np.memmap(path, dtype=dtype_in, mode="r")
                ix = torch.randint(len(data) - block_size, (micro_batch,))
                x = torch.stack(
                    [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
                )
                y = torch.stack(
                    [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
                )
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

        if "cuda" in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch, None


def build_lr_scheduler(optimizer, warmup_iters, max_iters, min_lr):
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    if warmup_iters > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1 / warmup_iters,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters]
        )
    return CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)
