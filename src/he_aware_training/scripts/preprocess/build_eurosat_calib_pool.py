import argparse
import glob
import json
import os

import numpy as np


def _image_norm(model_name):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    slug = "models--" + model_name.replace("/", "--")
    for h in glob.glob(os.path.join(hub, slug, "snapshots", "*", "preprocessor_config.json")):
        d = json.load(open(h))
        if "image_mean" in d:
            return d["image_mean"], d["image_std"]
    return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def _local_parquet(repo_id, split):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    slug = "datasets--" + repo_id.replace("/", "--")
    for f in glob.glob(os.path.join(hub, slug, "snapshots", "*", "**", "*.parquet"), recursive=True):
        if os.path.basename(f).startswith(split + "-"):
            return f
    raise SystemExit(f"no local {split} parquet for {repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="eurosat")   # must match the backend dataset cfg `name`
    ap.add_argument("--hf-path", default="tanganke/eurosat")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--n-images", type=int, default=512)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub.constants import HF_HOME

    mean = np.array(_image_norm(args.model)[0], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(_image_norm(args.model)[1], dtype=np.float32).reshape(3, 1, 1)
    pq = _local_parquet(args.hf_path, args.split)
    ds = load_dataset("parquet", data_files={args.split: pq})[args.split]
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_images, len(ds))))
    print(f"[pool] {len(ds)} imgs @ {args.res} from {os.path.basename(pq)}", flush=True)

    R = args.res
    out = np.empty((len(ds), 3, R, R), dtype=np.float32)
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB").resize((R, R))
        a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        out[i] = (a - mean) / std

    slug = args.model.split("/")[-1]
    pools = os.path.join(HF_HOME, "perseus", "pools")
    os.makedirs(pools, exist_ok=True)
    path = os.path.join(pools, f"{args.dataset_name}_{slug}_{args.n_images}.npy")
    np.save(path, out)
    print(f"[pool] wrote {out.shape} -> {path}")
    print("BUILD_EUROSAT_POOL_DONE")


if __name__ == "__main__":
    main()
