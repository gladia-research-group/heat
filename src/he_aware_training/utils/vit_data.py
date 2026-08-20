import os
import pickle
import time

import numpy as np
import torch


def compose_get_image_batch(model_name, data_path, dataset_name, device, micro_batch):
    data_dir = os.path.join(data_path, dataset_name)
    slug = model_name.split("/")[-1]

    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    C, H, W = meta["channels"], meta["height"], meta["width"]

    paths = {}
    for split in ("train", "val"):
        px = os.path.join(data_dir, f"{slug}_{split}_pixels.bin")
        lb = os.path.join(data_dir, f"{slug}_{split}_labels.bin")
        if not os.path.exists(px):
            raise FileNotFoundError(f"missing {px} — run prepare_eurosat_vit80.py first")
        paths[split] = (px, lb)

    def get_batch(split):
        px_path, lb_path = paths["train" if split == "train" else "val"]
        for attempt in range(5):   # transient Lustre EACCES/EIO on shared filesystems -> retry
            try:
                labels = np.memmap(lb_path, dtype=np.int32, mode="r")
                pixels = np.memmap(px_path, dtype=np.float16, mode="r",
                                   shape=(len(labels), C, H, W))
                ix = torch.randint(len(labels), (micro_batch,))
                x = torch.stack([torch.from_numpy(np.asarray(pixels[i], dtype=np.float32))
                                 for i in ix])
                y = torch.tensor([int(labels[i]) for i in ix], dtype=torch.long)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

        if "cuda" in str(device):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch, meta


def load_vit_for_heat(model_dir, num_labels, resolution=None):
    import functools

    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        model_dir, num_labels=int(num_labels), ignore_mismatched_sizes=True)
    model.config.vocab_size = int(num_labels)          # trainer resize-check no-ops
    if resolution:
        model.forward = functools.partial(model.forward, interpolate_pos_encoding=True)
    return model
