import argparse
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--dataset-name", default="EuroSAT_vit80")
    ap.add_argument("--hf-path", default="tanganke/eurosat")
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import glob
    import json as _json

    from datasets import load_dataset

    def _image_norm(model_name):
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        slug = "models--" + model_name.replace("/", "--")
        hits = glob.glob(os.path.join(hub, slug, "snapshots", "*", "preprocessor_config.json"))
        for h in hits:
            try:
                d = _json.load(open(h))
                if "image_mean" in d and "image_std" in d:
                    return d["image_mean"], d["image_std"], h
            except (OSError, ValueError):
                continue
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], "default(0.5)"

    im_mean, im_std, src = _image_norm(args.model)
    mean = np.array(im_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(im_std, dtype=np.float32).reshape(3, 1, 1)
    print(f"[prep] model={args.model} res={args.res} mean={im_mean} std={im_std}  (from {src})")

    def _local_parquet(repo_id):
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        slug = "datasets--" + repo_id.replace("/", "--")
        files = glob.glob(os.path.join(hub, slug, "snapshots", "*", "**", "*.parquet"),
                          recursive=True)
        pick = {}
        for f in files:
            base = os.path.basename(f)
            for split in ("train", "test", "validation"):
                if base.startswith(split + "-"):
                    pick.setdefault(split, f)
        return pick

    local = _local_parquet(args.hf_path)
    if local:
        print(f"[prep] loading LOCAL parquet: { {k: os.path.basename(v) for k, v in local.items()} }")
        ds = load_dataset("parquet", data_files=local)
    else:
        ds = load_dataset(args.hf_path)
    print(f"[prep] loaded {args.hf_path}: { {k: len(v) for k, v in ds.items()} }")

    # EuroSAT ships train/test; carve a val split from train if no explicit val.
    if "test" in ds and "train" in ds:
        splits = {"train": ds["train"], "val": ds["test"]}
    else:
        base = ds["train"].train_test_split(test_size=args.val_frac, seed=args.seed)
        splits = {"train": base["train"], "val": base["test"]}

    label_feat = splits["train"].features["label"]
    classes = list(getattr(label_feat, "names", []))
    num_classes = len(classes) or int(max(splits["train"]["label"])) + 1
    print(f"[prep] num_classes={num_classes} classes={classes}")

    out_dir = os.path.join(args.data_path, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    slug = args.model.split("/")[-1]
    R = args.res

    for split, dset in splits.items():
        n = len(dset)
        px_path = os.path.join(out_dir, f"{slug}_{split}_pixels.bin")
        lb_path = os.path.join(out_dir, f"{slug}_{split}_labels.bin")
        px = np.memmap(px_path, dtype=np.float16, mode="w+", shape=(n, 3, R, R))
        lb = np.memmap(lb_path, dtype=np.int32, mode="w+", shape=(n,))

        for i, ex in enumerate(dset):
            img = ex["image"].convert("RGB").resize((R, R))
            a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            px[i] = ((a - mean) / std).astype(np.float16)
            lb[i] = int(ex["label"])
            if (i + 1) % 2000 == 0:
                print(f"  [{split}] {i+1}/{n}", flush=True)

        px.flush(); lb.flush(); del px, lb
        print(f"[prep] {split}: {n} imgs -> {px_path}")

    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"channels": 3, "height": R, "width": R,
                     "num_classes": num_classes, "classes": classes,
                     "model": args.model, "resolution": R}, f)
    print(f"[prep] wrote meta.pkl -> {out_dir}")
    print("PREP_EUROSAT_DONE")


if __name__ == "__main__":
    main()
