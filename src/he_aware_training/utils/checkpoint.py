import os
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

from he_aware_training import PROJECT_ROOT


@dataclass
class ResumeState:
    iter_num: int = 0
    scheduler_state: Optional[dict] = field(default=None)
    warmup_on_resume: int = 0


def get_checkpoint_dir(cfg, run_id=None):
    """Get the checkpoint directory path for current config."""
    model_name_clean = cfg.model.name.split("/")[-1]
    if run_id:
        return os.path.join(PROJECT_ROOT, "checkpoints", model_name_clean, run_id)
    return os.path.join(PROJECT_ROOT, "checkpoints", model_name_clean)


def find_checkpoint(value, cfg):    
    """
    Resolve a checkpoint value to a path.

    Args:
        value options:
            - None/False: No checkpoint to load
            - 'auto'/'last'/True: Auto-find latest checkpoint
            - path string: Use explicit checkpoint path
        cfg: Configuration object containing checkpoint directory settings via get_checkpoint_dir()

    Returns:
        str or None: 
            - Valid checkpoint file path (typically ending with 'last.pt') if found
            - None if:
                - value is falsy
                - explicit path doesn't exist
                - base checkpoint directory doesn't exist
                - no checkpoints found in auto-search mode
                
    Behavior:
            - For explicit paths: Checks file existence and warns to console if not found
            - For auto mode: Searches all subdirectories in the checkpoint base directory
              for 'last.pt' files and returns the path with the most recent modification time
            - Skips non-directory entries when scanning for checkpoints
    """
    if not value:
        return None

    if value not in ["auto", "last", True]:
        if os.path.exists(value):
            return value
        print(f"Warning: Checkpoint not found at {value}")
        return None

    base_dir = get_checkpoint_dir(cfg)
    if not os.path.exists(base_dir):
        return None

    latest_ckpt = None
    latest_mtime = 0

    for run_dir in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        last_pt = os.path.join(run_path, "last.pt")
        if os.path.exists(last_pt):
            mtime = os.path.getmtime(last_pt)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_ckpt = last_pt

    return latest_ckpt


def find_latest_checkpoint(cfg):
    """Find the latest checkpoint based on cfg.trainer.resume_from."""
    return find_checkpoint(getattr(cfg.trainer, "resume_from", None), cfg)


def save_checkpoint(
    model, optimizer, iter_num, cfg, out_dir, ddp=False, filename=None, scheduler=None
):
    """Save a full training checkpoint."""
    raw_model = model.module if ddp else model

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter_num": iter_num,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    os.makedirs(out_dir, exist_ok=True)

    if filename:
        torch.save(checkpoint, os.path.join(out_dir, filename))
    torch.save(checkpoint, os.path.join(out_dir, "last.pt"))

    return out_dir


def _interpolate_pos_embed(old_embed, new_num_tokens):
    cls_tok = old_embed[:, :1, :]
    patch_embed = old_embed[:, 1:, :]

    old_n = patch_embed.shape[1]
    new_n = new_num_tokens - 1
    if old_n == new_n:
        return old_embed

    old_grid = int(old_n**0.5)
    new_grid = int(new_n**0.5)
    hidden = patch_embed.shape[2]

    patch_embed = (
        patch_embed.reshape(1, old_grid, old_grid, hidden).permute(0, 3, 1, 2).float()
    )
    patch_embed = F.interpolate(
        patch_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False
    )
    patch_embed = (
        patch_embed.permute(0, 2, 3, 1).reshape(1, new_n, hidden).to(cls_tok.dtype)
    )

    print(
        f"  [pos_embed] interpolated {old_n} -> {new_n} patches ({old_grid}x{old_grid} -> {new_grid}x{new_grid})"
    )
    return torch.cat([cls_tok, patch_embed], dim=1)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda", strict=True,
                    transpose_keys=()):
    """
    Load a checkpoint and restore model, optimizer, and scheduler states.
    
    This function loads a checkpoint from disk and restores the model's state dictionary.
    It handles various migration scenarios including:
    - Converting scalar tensor parameters to the expected shape
    - Interpolating position embeddings when model sequence length differs from checkpoint
    - Skipping incompatible layers and reinitializing them
    - Optionally restoring optimizer and scheduler states
    
    Args:
        checkpoint_path (str): Path to the checkpoint file to load.
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state for.
            If None, optimizer state is not loaded. Defaults to None.
        device (str, optional): Device to load the checkpoint on (e.g., "cuda" or "cpu").
            Defaults to "cuda".
        strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint
            match the model's state dict. Defaults to True.
    
    Returns:
        tuple: A tuple containing:
            - iter_num (int): The iteration number from checkpoint, or 0 for old-format checkpoints.
            - scheduler_state (dict or None): The scheduler state dictionary if present in checkpoint,
              otherwise None.
    
    Note:
        - Optimizer state is skipped if any state dict migrations occur to allow Adam to reinitialize.
        - Old-format checkpoints (containing only weights) are supported and start from iteration 0.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    migrated = False
    for key in list(state_dict.keys()):
        if any(key.endswith(s) for s in transpose_keys) and key in model_state:
            t = state_dict[key]
            if t.dim() == 2 and t.shape[0] == t.shape[1]:
                print(f"  [transpose-square] {key} (caller-declared Linear->Conv1D)")
                state_dict[key] = t.t().contiguous()
                migrated = True
    for key in list(state_dict.keys()):
        if key.endswith("gk_scale") and key in model_state:
            if state_dict[key].dim() == 0 and model_state[key].dim() > 0:
                print(f"  [migrate] {key}: scalar -> {tuple(model_state[key].shape)}")
                state_dict[key] = state_dict[key].expand(model_state[key].shape).clone()
                migrated = True

    pos_key = "vit.embeddings.position_embeddings"
    if pos_key in state_dict and pos_key in model_state:
        ckpt_len = state_dict[pos_key].shape[1]
        model_len = model_state[pos_key].shape[1]
        if ckpt_len != model_len:
            state_dict[pos_key] = _interpolate_pos_embed(state_dict[pos_key], model_len)
            migrated = True

    for key in list(state_dict.keys()):
        if key in model_state and state_dict[key].shape != model_state[key].shape:
            # Conv1D (HF GPT-2) <-> Linear (EncLLM surgery): weights are transposed.
            ckpt_t = state_dict[key]
            model_t = model_state[key]
            if ckpt_t.dim() == 2 and tuple(ckpt_t.shape) == tuple(model_t.shape)[::-1]:
                print(
                    f"  [transpose] {key}: {tuple(ckpt_t.shape)} -> {tuple(model_t.shape)}"
                )
                state_dict[key] = ckpt_t.t().contiguous()
                migrated = True
                continue
            print(
                f"  [skip] {key}: checkpoint shape {tuple(state_dict[key].shape)} != model shape {tuple(model_state[key].shape)}, reinitializing"
            )
            del state_dict[key]
            migrated = True

    if "model_state_dict" in checkpoint:
        model.load_state_dict(state_dict, strict=strict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            if migrated:
                print(
                    "  [migrate] Skipping optimizer state (shapes changed, Adam will reinitialize)"
                )
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        iter_num = checkpoint.get("iter_num", 0)
        print(f"Resumed from iteration {iter_num}")
    else:
        model.load_state_dict(state_dict, strict=strict)
        iter_num = 0
        print("Loaded old-format checkpoint (model weights only), starting from iter 0")

    scheduler_state = checkpoint.get("scheduler_state_dict", None)
    return iter_num, scheduler_state


def load_init_checkpoint(model, cfg, device, ddp=False, master_process=True,
                         transpose_keys=()):
    checkpoint_path = find_checkpoint(getattr(cfg.trainer, "init_from", None), cfg)
    if not checkpoint_path:
        return

    raw_model = model.module if ddp else model
    if master_process:
        print(f"Loading initial weights from: {checkpoint_path}")
    load_checkpoint(
        checkpoint_path, raw_model, optimizer=None, device=device, strict=False,
        transpose_keys=transpose_keys,
    )  # return value ignored


def resume_training(model, optimizer, cfg, device, ddp=False, master_process=True):
    state = ResumeState()
    warmup_on_resume = getattr(cfg.trainer, "warmup_on_resume", 0)
    reset_lr = getattr(cfg.trainer, "reset_lr_on_resume", False)
    max_iters = getattr(cfg.trainer, "max_iters", 0)
    max_lr = cfg.trainer.learning_rate

    checkpoint_path = find_latest_checkpoint(cfg)
    if not checkpoint_path:
        return state

    raw_model = model.module if ddp else model
    iter_num, sched_state = load_checkpoint(
        checkpoint_path, raw_model, optimizer, device
    )

    state.iter_num = iter_num

    if reset_lr:
        state.scheduler_state = None
        if master_process:
            remaining = max_iters - iter_num
            print(
                f"LR reset: cosine from {max_lr:.1e} over {remaining} remaining steps"
            )
    elif warmup_on_resume > 0:
        state.scheduler_state = None
        state.warmup_on_resume = warmup_on_resume
        if master_process:
            print(f"Re-warmup: {warmup_on_resume} steps starting at iter {iter_num}")
    else:
        state.scheduler_state = sched_state

    if master_process:
        print(f"Resuming training from iteration {iter_num}")

    return state


def save_eval_checkpoint(model, optimizer, iter_num, cfg, ddp=False, scheduler=None):
    """Save checkpoint at eval step, using wandb run_id for the directory."""
    run_id = wandb.run.id if wandb.run else "manual_run"
    out_dir = get_checkpoint_dir(cfg, run_id)
    save_checkpoint(
        model,
        optimizer,
        iter_num,
        cfg,
        out_dir,
        ddp=ddp,
        filename=f"ckpt_{iter_num}.pt",
        scheduler=scheduler,
    )
    print(f"Saved checkpoint to: {out_dir}")
