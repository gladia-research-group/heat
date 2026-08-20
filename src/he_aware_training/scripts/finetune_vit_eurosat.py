import argparse
import os
import pickle
import time

import numpy as np
import torch

from he_aware_training.utils.vit_data import compose_get_image_batch

ROOT = os.environ.get("PROJECT_ROOT", ".")


def evaluate(model, data_dir, slug, C, H, W, device, batch=128):
    """Full pass over the val split (deterministic, unlike the random train sampler)."""
    lb = np.memmap(os.path.join(data_dir, f"{slug}_val_labels.bin"), dtype=np.int32, mode="r")
    px = np.memmap(os.path.join(data_dir, f"{slug}_val_pixels.bin"), dtype=np.float16,
                   mode="r", shape=(len(lb), C, H, W))
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for i in range(0, len(lb), batch):
            x = torch.from_numpy(np.asarray(px[i:i + batch], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.asarray(lb[i:i + batch], dtype=np.int64)).to(device)
            logits = model(pixel_values=x, interpolate_pos_encoding=True).logits
            correct += (logits.argmax(-1) == y).sum().item()
            tot += len(y)
    model.train()
    return correct / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=f"{ROOT}/data")
    ap.add_argument("--dataset-name", default="EuroSAT_vit80")
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--out-dir", default=f"{ROOT}/checkpoints/vit/eurosat_vit80")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()

    from transformers import ViTForImageClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(args.data_path, args.dataset_name)
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    C, H, W = meta["channels"], meta["height"], meta["width"]
    n_cls = meta["num_classes"]
    slug = args.model.split("/")[-1]
    print(f"[ft] device={device} res={H} classes={n_cls} data={data_dir}", flush=True)

    get_batch, _ = compose_get_image_batch(args.model, args.data_path, args.dataset_name,
                                           device, args.batch_size)

    model = ViTForImageClassification.from_pretrained(
        args.model, num_labels=n_cls, ignore_mismatched_sizes=True).to(device)
    model.train()

    n_train = len(np.memmap(os.path.join(data_dir, f"{slug}_train_labels.bin"),
                            dtype=np.int32, mode="r"))
    steps_per_epoch = max(1, n_train // args.batch_size)
    max_iters = steps_per_epoch * args.epochs
    warmup = int(args.warmup_frac * max_iters)
    print(f"[ft] n_train={n_train} steps/epoch={steps_per_epoch} max_iters={max_iters}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max_iters,
        pct_start=max(args.warmup_frac, 1e-3), anneal_strategy="cos")

    os.makedirs(args.out_dir, exist_ok=True)
    best, t0 = 0.0, time.time()
    for it in range(1, max_iters + 1):
        x, y = get_batch("train")
        out = model(pixel_values=x, labels=y, interpolate_pos_encoding=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if it % 50 == 0:
            print(f"[ft] iter {it}/{max_iters} loss {out.loss.item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if it % args.eval_every == 0 or it == max_iters:
            acc = evaluate(model, data_dir, slug, C, H, W, device)
            print(f"[ft] iter {it} VAL_ACC {acc:.4f}", flush=True)
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "val_acc": acc, "iter": it,
                            "num_classes": n_cls, "resolution": H, "base_model": args.model},
                           os.path.join(args.out_dir, "last.pt"))
                model.save_pretrained(os.path.join(args.out_dir, "hf"))
                print(f"[ft] saved (best {best:.4f}) -> {args.out_dir}", flush=True)

    print(f"[ft] BEST_VAL_ACC {best:.4f}")
    print("FT_VIT_EUROSAT_DONE")


if __name__ == "__main__":
    main()
