import argparse
import os

def _calib_file(calib):
    """Accept a dir (merged configs.json inside, or the retired thor/ln/gelu split) or a file."""
    import os
    if os.path.isfile(calib):
        return calib
    merged = os.path.join(calib, "configs.json")
    if os.path.isfile(merged):
        return merged
    raise SystemExit(f"no configs.json in {calib} (and the split form has been retired)")


import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="heat ckpt to load; omit to evaluate the config's own weights "
                         "(e.g. a squeezed HF backbone with surgery overridden off)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--data", default="data/EuroSAT_vit80")
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--counts-only", action="store_true",
                    help="report the learned circuit only; skip the val accuracy sweep")
    ap.add_argument("--allow-partial", action="store_true",
                    help="tolerate missing/unexpected keys on ckpt load (default: hard fail — "
                         "a nonzero count means a wrong CKPT<->CALIB pairing)")
    args, extra_overrides = ap.parse_known_args()

    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from he_aware_training.utils.surgery import perform_he_surgery

    root = os.environ["PROJECT_ROOT"]
    with initialize_config_dir(version_base=None, config_dir=f"{root}/configs"):
        cfg = compose(config_name="he_aware_train_vit", overrides=[
            "device=cpu",
            f"model.approximation.hybrid.thor.calib_path={_calib_file(args.calib)}",
            f"model.approximation.hybrid.ln.calib_path={_calib_file(args.calib)}",
            f"model.approximation.hybrid.gelu.calib_path={_calib_file(args.calib)}",
            "model.approximation.hybrid.thor.squeeze_bound=30",
        ] + extra_overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model.instance)
    model = perform_he_surgery(model, cfg, verbose=False)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        live = set(n for n, _ in model.named_parameters())
        if any(".layers." in n for n in live) and any(".encoder.layer." in k for k in sd):
            sd = {k.replace(".encoder.layer.", ".layers.")
                   .replace(".intermediate.dense.", ".mlp.fc1.")
                   .replace(".output.dense.", ".mlp.fc2."): v for k, v in sd.items()}
        elif any(".encoder.layer." in n for n in live) and any(".layers." in k for k in sd):
            sd = {k.replace(".layers.", ".encoder.layer.")
                   .replace(".mlp.fc1.", ".intermediate.dense.")
                   .replace(".mlp.fc2.", ".output.dense."): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[eval] loaded {args.ckpt}: missing={len(miss)} unexpected={len(unexp)}")
        if (miss or unexp) and not args.allow_partial:
            raise SystemExit(f"[eval] CKPT<->CALIB pairing mismatch (missing {miss[:3]}... "
                             f"unexpected {unexp[:3]}...) — wrong calib for this ckpt? "
                             f"Pass --allow-partial to proceed anyway.")
    else:
        print("[eval] no --ckpt: evaluating the config's own weights as loaded")
    model.to(device).eval()

    slug = args.model.split("/")[-1]
    lb = np.memmap(f"{args.data}/{slug}_val_labels.bin", dtype=np.int32, mode="r")
    px = np.memmap(f"{args.data}/{slug}_val_pixels.bin", dtype=np.float16, mode="r",
                   shape=(len(lb), 3, 80, 80))
    from he_aware_training.utils.loss import DepthRegularizer
    depth_reg = DepthRegularizer(model, lambda_depth=0.0, scheduler=None)
    with torch.no_grad():
        x0 = torch.from_numpy(np.asarray(px[:2], dtype=np.float32)).to(device)
        model(pixel_values=x0, interpolate_pos_encoding=True)
    dstats = depth_reg.get_eval_stats()
    print("\n[counts] learned per-site iterations (mode = executed/deploy rule; hard = 95%-mass diagnostic):")
    for k in sorted(k for k in dstats if k.startswith("depth_mode/")):
        print(f"   {k[len('depth_mode/'):]:70s} {dstats[k]}")
    for k in sorted(k for k in dstats if k.startswith("depth_hard/")):
        print(f"   {k[len('depth_hard/'):]:70s} {dstats[k]}")
    for k in ("depth/global_avg_hard", "depth/global_avg_mode", "depth/global_avg_mass"):
        if k in dstats:
            print(f"[counts] {k} = {dstats[k]:.4f}")
    ln_gs = [v for k, v in dstats.items()
             if k.startswith("depth_mode/") and "inv_sqrt" in k and "goldschmidt" in k]
    sm_ref = [v for k, v in dstats.items()
              if k.startswith("depth_mode/") and "softmax" in k and "refine" in k]
    if ln_gs:
        print(f"[counts] LN goldschmidt: sum={sum(ln_gs)} min={min(ln_gs)} max={max(ln_gs)} n={len(ln_gs)}")
    if sm_ref:
        print(f"[counts] softmax refine: sum={sum(sm_ref)} min={min(sm_ref)} max={max(sm_ref)} n={len(sm_ref)}")
    depth_reg.close()
    if args.counts_only:
        print("EVAL_VIT_HEAT_ACC_DONE")
        return

    correct = tot = nonfinite = 0
    with torch.no_grad():
        for i in range(0, len(lb), args.batch):
            x = torch.from_numpy(np.asarray(px[i:i+args.batch], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.asarray(lb[i:i+args.batch], dtype=np.int64)).to(device)
            logits = model(pixel_values=x, interpolate_pos_encoding=True).logits
            nonfinite += int((~torch.isfinite(logits).all(-1)).sum())
            pred = logits.argmax(-1)
            correct += int((pred == y).sum()); tot += len(y)
    print(f"[eval] HEAT deploy-mode EuroSAT val: top1 = {correct/tot:.4f} "
          f"({correct}/{tot})  non-finite logit rows: {nonfinite}/{tot}")
    print("EVAL_VIT_HEAT_ACC_DONE")


if __name__ == "__main__":
    main()
