import os
import pickle

import numpy as np
import torch

check = False

_image_meta_cache = {}


def get_image_batch(cfg, split):
    model_name = cfg.model.name.split("/")[-1]
    data_dir = os.path.join(cfg.misc.data_path, cfg.dataset.name)

    pixel_path = os.path.join(data_dir, f"{model_name}_{split}_pixels.bin")
    label_path = os.path.join(data_dir, f"{model_name}_{split}_labels.bin")

    if not os.path.exists(pixel_path):
        return None, None

    cache_key = data_dir
    if cache_key not in _image_meta_cache:
        meta_path = os.path.join(data_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            _image_meta_cache[cache_key] = pickle.load(f)

    meta = _image_meta_cache[cache_key]
    C, H, W = meta["channels"], meta["height"], meta["width"]
    image_size = C * H * W

    pixels = np.memmap(pixel_path, dtype=np.float16, mode="r")
    labels = np.memmap(label_path, dtype=np.int32, mode="r")
    num_images = len(labels)

    batch_size = (
        cfg.trainer.batch_size if hasattr(cfg, "trainer") else cfg.model.batch_size
    )
    ix = torch.randint(num_images, (batch_size,))

    x = torch.stack(
        [
            torch.from_numpy(pixels[i * image_size : (i + 1) * image_size].copy())
            .reshape(C, H, W)
            .float()
            for i in ix
        ]
    )

    y = torch.tensor([labels[i] for i in ix], dtype=torch.long)

    if "cuda" in cfg.device:
        x = x.pin_memory().to(cfg.device, non_blocking=True)
        y = y.pin_memory().to(cfg.device, non_blocking=True)
    else:
        x, y = x.to(cfg.device), y.to(cfg.device)

    return x, y


def get_batch(cfg, split):
    global check
    model_name = cfg.model.name.split("/")[-1]
    filename = f"{model_name}_{split}.bin"
    if check:
        print("loaded data from:", filename)
        check = False
    dtype_in = np.uint32 if cfg.model.data_dtype == "uint32" else np.uint16
    data_dir = os.path.join(cfg.misc.data_path, cfg.dataset.name)
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None, None
    data = np.memmap(path, dtype=dtype_in, mode="r")

    batch_size = (
        cfg.trainer.batch_size if hasattr(cfg, "trainer") else cfg.model.batch_size
    )
    block_size = cfg.model.block_size

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    if "cuda" in cfg.device:
        x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(
            cfg.device, non_blocking=True
        )
    else:
        x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y
