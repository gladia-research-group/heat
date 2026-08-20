import os
import time
import math
import gc
import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb

from he_aware_training import PROJECT_ROOT
from he_aware_training.modules.learnable_components import PonderApproximation
from he_aware_training.utils.surgery import audit_surgery, perform_he_surgery
from he_aware_training.utils.utils import pop_squeeze_stats
from he_aware_training.utils.checkpoint import (
    load_init_checkpoint,
    resume_training,
    save_eval_checkpoint,
)
from he_aware_training.utils.train import (
    build_lr_scheduler,
    build_run_tags,
    estimate_loss,
    freeze_model_components,
)


def compute_main_loss(outputs, X, teacher, distill_T):
    if teacher is None:
        return outputs.loss if hasattr(outputs, "loss") else outputs[1]
    with torch.no_grad():
        t_logits = teacher(X).logits
    V = outputs.logits.shape[-1]
    return F.kl_div(
        F.log_softmax(outputs.logits.float().reshape(-1, V) / distill_T, dim=-1),
        F.softmax(t_logits.float().reshape(-1, V) / distill_T, dim=-1),
        reduction="batchmean",
    ) * (distill_T * distill_T)


def set_soft_training(model, enabled):
    for module in model.modules():
        if isinstance(module, PonderApproximation):
            module.soft_training = enabled


def apply_halt_ramp(model, slope):
    if slope is None:
        return
    n = 0
    for m in model.modules():
        if not isinstance(m, PonderApproximation) or m.halt_logits.numel() == 0:
            continue
        with torch.no_grad():
            idx = torch.arange(m.halt_logits.numel(), device=m.halt_logits.device,
                               dtype=m.halt_logits.dtype)
            m.halt_logits.copy_(slope * (idx - m.optimal_idx))
            if m.min_iter > 0:
                m.halt_logits[:m.min_iter] = -10.0
        n += 1
    print(f"[halt-ramp] monotone ramp init slope={slope} applied to {n} ponder sites")


OPERATOR_SUFFIXES = (".p", ".q", ".lin_alpha", ".lin_beta",
                     ".init_alpha", ".init_beta", ".refine_alpha", ".refine_beta")


def setup_optimizer(model, cfg):
    ponder_params = []
    operator_params = []
    activation_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "halt_logits" in name:
            ponder_params.append(p)
        elif name.endswith(OPERATOR_SUFFIXES):
            operator_params.append(p)
        elif "scale" in name or "alpha" in name:
            activation_params.append(p)
        else:
            other_params.append(p)

    lr_groups = cfg.trainer.lr_groups
    optim_groups = [
        {"params": other_params, "lr": lr_groups.learning_rate, "name": "task"},
        {
            "params": activation_params,
            "lr": lr_groups.activation_lr,
            "name": "activation",
        },
        {"params": ponder_params, "lr": lr_groups.ponder_lr, "name": "ponder"},
        {"params": operator_params, "lr": lr_groups.operator_lr, "name": "operator"},
    ]

    for g in optim_groups:
        g["initial_lr"] = g["lr"]

    print("[Optimizer] Parameter groups:")
    print(f"   task:       {len(other_params)} params, lr={lr_groups.learning_rate}")
    print(
        f"   activation: {len(activation_params)} params, lr={lr_groups.activation_lr}"
    )
    print(f"   ponder:     {len(ponder_params)} params, lr={lr_groups.ponder_lr}")
    print(f"   operator:   {len(operator_params)} params, lr={lr_groups.operator_lr}")

    optimizer_factory = instantiate(cfg.trainer.optimizer, _partial_=True)
    optimizer = optimizer_factory(params=optim_groups)

    return optimizer


def _flush_cuda_allocator():
    """Release Python-unreferenced tensors and return cached blocks to CUDA driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(
    version_base=None,
    config_path=f"{PROJECT_ROOT}/configs",
    config_name="he_aware_train",
)
def main(cfg: DictConfig):
    device = cfg.device

    torch.manual_seed(cfg.get("seed", 1337))
    torch.backends.cuda.matmul.allow_tf32 = True

    run_tags = list(cfg.core.tags) + build_run_tags(cfg)
    wandb.init(
        entity=cfg.core.entity,
        project=cfg.core.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=run_tags,
    )

    block_size = cfg.model.block_size
    micro_batch = cfg.trainer.batch_size
    target_tokens = cfg.trainer.target_tokens_per_update
    tokens_per_micro_step = block_size * micro_batch
    grad_accum_steps = max(1, int(math.ceil(target_tokens / tokens_per_micro_step)))

    min_lr = cfg.trainer.min_lr
    max_iters = cfg.trainer.max_iters

    print("\n" + "=" * 40)
    print("TRAINING CONFIG")
    print("=" * 40)
    print(f"   Model:             {cfg.model.name}")
    print(
        f"   Surgery:           Norm={cfg.model.surgery.norm}, Act={cfg.model.surgery.activation}, Attn={cfg.model.surgery.attention}"
    )
    print("=" * 40 + "\n")

    get_batch, _ = instantiate(cfg.dataset.get_batch)

    print("Instantiating model...")
    model = instantiate(cfg.model.instance)

    if model.config.vocab_size != cfg.model.vocab_size:
        model.resize_token_embeddings(cfg.model.vocab_size)

    model = perform_he_surgery(model, cfg, verbose=True)
    audit_surgery(model, cfg)
    if not cfg.model.surgery.activation:
        from he_aware_training.utils.surgery import install_gelu_input_guards
        approx = cfg.model.approximation[cfg.model.approximation.name]
        gelu_ns = "gelu" if approx.get("gelu_kind", "sign_poly") == "thor" else "softgelu"
        install_gelu_input_guards(model, approx[gelu_ns].calib_path, component=gelu_ns)
    _flush_cuda_allocator()
    load_init_checkpoint(model, cfg, device)
    model.to(device)
    apply_halt_ramp(model, cfg.trainer.halt_init_slope)

    model = freeze_model_components(model, cfg)

    model.train()

    distill_T = cfg.trainer.distill_temperature
    teacher = None
    if cfg.trainer.distill:
        tdir = cfg.trainer.teacher_dir
        teacher = instantiate(cfg.model.instance, model_dir=tdir) if tdir \
            else instantiate(cfg.model.instance)
        if teacher.config.vocab_size != cfg.model.vocab_size:
            teacher.resize_token_embeddings(cfg.model.vocab_size)
        teacher.to(device).eval().requires_grad_(False)
        print(f"[distill] teacher = {tdir or 'model.instance (init ckpt)'} — KL(teacher‖student), T={distill_T}")

    regularizer = instantiate(cfg.trainer.regularizer, model=model)
    distribution_scheduler = instantiate(cfg.trainer.scheduler)
    depth_regularizer = instantiate(
        cfg.trainer.depth_regularizer, model=model, scheduler=distribution_scheduler
    )

    optimizer = setup_optimizer(model, cfg)

    scheduler_factory = instantiate(cfg.trainer.lr_scheduler, _partial_=True)
    lr_sched = scheduler_factory(optimizer)

    rs = resume_training(model, optimizer, cfg, device, ddp=False)
    iter_num = rs.iter_num

    if rs.warmup_on_resume > 0 and rs.iter_num > 0:
        remaining = max_iters - rs.iter_num
        lr_sched = build_lr_scheduler(optimizer, rs.warmup_on_resume, remaining, min_lr)
    elif getattr(cfg.trainer, "reset_lr_on_resume", False) and rs.iter_num > 0:
        remaining = max_iters - rs.iter_num
        lr_sched = build_lr_scheduler(optimizer, 0, remaining, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"]
    elif rs.scheduler_state is not None:
        lr_sched.load_state_dict(rs.scheduler_state)

    soft_training_warmup = cfg.trainer.soft_training_warmup
    depth_reg_warmup = cfg.trainer.depth_reg_warmup
    act_reg_warmup = cfg.trainer.act_reg_warmup
    if act_reg_warmup > 0:
        print(f"[Warmup] Squeeze/activation penalty ramps linearly 0→full over {act_reg_warmup} steps")
    if soft_training_warmup > 0:
        set_soft_training(model, False)
        print(
            f"[Warmup] Starting with deterministic PonderNet for {soft_training_warmup} steps"
        )
        print(
            f"[Warmup] Depth reg will ramp linearly over {depth_reg_warmup} steps after phase switch"
        )

    t0 = time.time()
    gnorm_ema = None
    halt_temp_end = cfg.trainer.halt_temp_end
    ponder_mods = [m for m in model.modules() if isinstance(m, PonderApproximation)]

    def _set_halt_temp(it):
        if halt_temp_end is None:
            return
        span = max(1, cfg.trainer.max_iters - soft_training_warmup)
        frac = min(1.0, max(0.0, (it - soft_training_warmup) / span))
        temp = 1.0 + frac * (halt_temp_end - 1.0)
        for m in ponder_mods:
            m.halt_temp = temp

    while iter_num <= cfg.trainer.max_iters:
        _set_halt_temp(iter_num)

        if iter_num == soft_training_warmup:   # warmup=0 ⇒ enter phase-2 (soft+phase2_lr) at iter 0
            set_soft_training(model, True)
            phase2_lr = cfg.trainer.phase2_lr
            for g in optimizer.param_groups:
                if g["name"] in ("activation", "task"):
                    g["lr"] = g["initial_lr"] = phase2_lr
            remaining = max_iters - iter_num
            lr_sched = build_lr_scheduler(optimizer, 0, remaining, min_lr)
            print(
                f"[Phase switch] Step {iter_num}: Weighted PonderNet, act & task lr -> {phase2_lr}"
            )

        is_depth_log_step = iter_num % 25 == 0
        is_eval_step = iter_num % cfg.trainer.eval_interval == 0

        if is_depth_log_step or is_eval_step:
            ponder_stats = depth_regularizer.get_eval_stats()
            wandb.log({"iter": iter_num, **ponder_stats})

        if is_eval_step:
            losses = estimate_loss(model, get_batch, cfg.trainer.eval_iters, device, teacher=teacher)
            vt = losses.get("val_teacher")
            t_str = f"  (FP teacher val {vt:.4f}, Δ {losses['val'] - vt:+.4f})" if vt is not None else ""
            nf = losses["train_nonfinite"] + losses["val_nonfinite"]
            nf_str = f"  [nf batches: train {losses['train_nonfinite']}, val {losses['val_nonfinite']} /{cfg.trainer.eval_iters}]" if nf else ""
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{t_str}{nf_str}"
            )

            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "train/nonfinite_batches": losses["train_nonfinite"],
                    "val/nonfinite_batches": losses["val_nonfinite"],
                    **({"val/teacher_loss": losses["val_teacher"],
                        "val/gap_vs_teacher": losses["val"] - losses["val_teacher"]}
                       if "val_teacher" in losses else {}),
                    "lr/task": optimizer.param_groups[0]["lr"],
                    "lr/activation": optimizer.param_groups[1]["lr"],
                    "lr/ponder": optimizer.param_groups[2]["lr"],
                }
            )

            if iter_num > 0:
                save_eval_checkpoint(
                    model, optimizer, iter_num, cfg, ddp=False, scheduler=lr_sched
                )

        optimizer.zero_grad(set_to_none=True)
        accum_task_loss = 0.0
        accum_act_reg = 0.0
        accum_depth_reg = 0.0
        in_warmup = soft_training_warmup > 0 and iter_num < soft_training_warmup
        for micro_step in range(grad_accum_steps):
            X, Y = get_batch("train")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(X, labels=Y)
                task_loss = compute_main_loss(outputs, X, teacher, distill_T)
                act_reg_loss = regularizer.pop_loss()
                if act_reg_warmup > 0 and iter_num < act_reg_warmup:
                    act_reg_loss = act_reg_loss * (iter_num / act_reg_warmup)

                if not in_warmup:
                    depth_kl = depth_regularizer.pop_loss(iter_num=iter_num)
                    depth_elapsed = iter_num - soft_training_warmup
                    if depth_reg_warmup > 0 and depth_elapsed < depth_reg_warmup:
                        depth_kl = depth_kl * (depth_elapsed / depth_reg_warmup)
                else:
                    depth_kl = torch.tensor(0.0, device=device)

                loss = (task_loss + act_reg_loss + depth_kl) / grad_accum_steps

            accum_task_loss += task_loss.item() / grad_accum_steps
            accum_act_reg += act_reg_loss.item() / grad_accum_steps
            accum_depth_reg += depth_kl.item() / grad_accum_steps
            loss.backward()

        bad_grad_params = 0
        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    bad_grad_params += 1
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if bad_grad_params > 0:
            print(f"[WARN] iter {iter_num}: NaN/Inf grads in {bad_grad_params} param tensors — scrubbed to 0")

        if cfg.trainer.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.trainer.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

        rel = cfg.trainer.grad_skip_rel
        rel_cut = max(rel * gnorm_ema, 10.0) if (rel > 0 and gnorm_ema is not None) else float("inf")
        if (bad_grad_params > 0 or not torch.isfinite(grad_norm)
                or grad_norm > cfg.trainer.grad_skip_threshold
                or grad_norm > rel_cut):
            print(f"[WARN] iter {iter_num}: gnorm {grad_norm:.3e} anomalous — update skipped")
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            if rel > 0:
                gnorm_ema = float(grad_norm) if gnorm_ema is None else 0.99 * gnorm_ema + 0.01 * float(grad_norm)
        lr_sched.step()
        if cfg.trainer.ponder_lr_const is not None:
            optimizer.param_groups[2]["lr"] = cfg.trainer.ponder_lr_const

        if iter_num % cfg.trainer.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            lr_task = optimizer.param_groups[0]["lr"]
            lr_act = optimizer.param_groups[1]["lr"]
            lr_ponder = optimizer.param_groups[2]["lr"]

            print(
                f"iter {iter_num}: task {accum_task_loss:.4f} | act_reg {accum_act_reg:.4f} | depth_reg {accum_depth_reg:.4f} | gnorm {grad_norm:.2f} | lr [t={lr_task:.1e} a={lr_act:.1e} p={lr_ponder:.1e}] | {dt*1000:.0f}ms"
            )

            act_stats = regularizer.get_stats_and_reset()
            depth_stats = depth_regularizer.get_stats_and_reset()

            squeeze_stats = {}
            if iter_num % 50 == 0:
                for stag, (rate, mex) in pop_squeeze_stats().items():
                    squeeze_stats[f"squeeze_viol/{stag}_rate"] = rate
                    squeeze_stats[f"squeeze_viol/{stag}_max_excess"] = mex
                viol = {k: v for k, v in squeeze_stats.items() if k.endswith("_rate") and v > 0}
                if viol:
                    print("[squeeze] " + " | ".join(
                        f"{k.split('/')[1][:-5]} {v:.2e}" for k, v in sorted(viol.items())))

            wandb.log(
                {
                    "train/batch_loss": accum_task_loss,
                    "train/act_reg_loss": accum_act_reg,
                    "train/depth_reg_loss": accum_depth_reg,
                    "train/total_loss": accum_task_loss
                    + accum_act_reg
                    + accum_depth_reg,
                    "train/grad_norm": grad_norm.item(),
                    **act_stats,
                    **depth_stats,
                    **squeeze_stats,
                },
                commit=False,
            )

        iter_num += 1

    # Cooldown: freeze ponder, hard sampling, task-only loss
    cooldown_iters = cfg.trainer.cooldown_iters
    if cooldown_iters > 0:
        set_soft_training(model, False)
        for p in [p for n, p in model.named_parameters() if "halt_logits" in n]:
            p.requires_grad = False
        for g in optimizer.param_groups:
            if g["name"] == "ponder":
                g["lr"] = g["initial_lr"] = 0.0

        print(f"\n[Cooldown] {cooldown_iters} steps, hard iterations, task-only")
        cooldown_end = iter_num + cooldown_iters
        t0 = time.time()

        while iter_num < cooldown_end:
            if iter_num % cfg.trainer.eval_interval == 0:
                losses = estimate_loss(model, get_batch, cfg.trainer.eval_iters, device)
                nf = losses["train_nonfinite"] + losses["val_nonfinite"]
                nf_str = f"  [nf batches: train {losses['train_nonfinite']}, val {losses['val_nonfinite']} /{cfg.trainer.eval_iters}]" if nf else ""
                print(
                    f"[Cooldown] step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}{nf_str}"
                )
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "train/nonfinite_batches": losses["train_nonfinite"],
                        "val/nonfinite_batches": losses["val_nonfinite"],
                    }
                )
                save_eval_checkpoint(
                    model, optimizer, iter_num, cfg, ddp=False, scheduler=lr_sched
                )

            optimizer.zero_grad(set_to_none=True)
            accum_task_loss = 0.0
            for micro_step in range(grad_accum_steps):
                X, Y = get_batch("train")
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(X, labels=Y)
                    task_loss = compute_main_loss(outputs, X, teacher, distill_T)
                    act_reg_loss = regularizer.pop_loss()
                    loss = (task_loss + act_reg_loss) / grad_accum_steps
                accum_task_loss += task_loss.item() / grad_accum_steps
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.trainer.grad_clip or float("inf")
            )
            if not torch.isfinite(grad_norm) or grad_norm > cfg.trainer.grad_skip_threshold:
                print(f"[WARN] iter {iter_num}: gnorm {grad_norm:.3e} anomalous — update skipped")
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()

            if iter_num % cfg.trainer.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(
                    f"[CD] iter {iter_num}: task {accum_task_loss:.4f} | gnorm {grad_norm:.2f} | {dt*1000:.0f}ms"
                )
                wandb.log(
                    {
                        "train/batch_loss": accum_task_loss,
                        "train/grad_norm": grad_norm.item(),
                    },
                    commit=False,
                )

            iter_num += 1

        losses = estimate_loss(model, get_batch, cfg.trainer.eval_iters, device)
        nf = losses["train_nonfinite"] + losses["val_nonfinite"]
        nf_str = f"  [nf batches: train {losses['train_nonfinite']}, val {losses['val_nonfinite']} /{cfg.trainer.eval_iters}]" if nf else ""
        print(f"[Cooldown] Final: train {losses['train']:.4f}, val {losses['val']:.4f}{nf_str}")
        save_eval_checkpoint(model, optimizer, iter_num, cfg, ddp=False)

    regularizer.close()
    depth_regularizer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
