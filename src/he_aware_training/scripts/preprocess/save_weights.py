#!/usr/bin/env python3
import json
import os
import zipfile

import tqdm
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

from he_aware_training import PROJECT_ROOT

@hydra.main(
    version_base=None,
    config_path=f"{PROJECT_ROOT}/configs",
    config_name="lm_eval",
)
def main(cfg: DictConfig):
    output_dir = cfg.eval.output_dir
    ckpt_path  = cfg.eval.backbone_ckpt

    print("Building classic GPT-2 (no surgery)...")
    model: AutoModelForCausalLM     = instantiate(cfg.model.instance)
    if model.config.vocab_size != cfg.model.vocab_size:
        model.resize_token_embeddings(cfg.model.vocab_size)
    model_tag = "classic"
    model.eval()

    out_dir = os.path.join(output_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}\n")

    modules_by_name = dict(model.named_modules())

    manifest = {"model": model_tag, "tensors": []}

    zip_path = os.path.join(out_dir, "weights.bin.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, (name, param) in enumerate(
            pbar := tqdm.tqdm(model.named_parameters(), desc="Saving weights")
        ):
            pbar.set_postfix_str(name)
            arr = param.detach().cpu().numpy()

            module_name, _, param_name = name.rpartition(".")
            parent = modules_by_name.get(module_name)
            if isinstance(parent, Conv1D) and param_name == "weight":
                arr = arr.T

            arr = np.ascontiguousarray(arr)
            tensor_path = f"tensors/{index:05d}.bin"
            zf.writestr(tensor_path, arr.tobytes(order="C"))
            manifest["tensors"].append(
                {
                    "name": name,
                    "path": tensor_path,
                    "shape": list(arr.shape),
                    "dtype": arr.dtype.str,
                }
            )
            print(f"Wrote {name} {arr.shape}")

        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nZipped -> {zip_path}")


if __name__ == "__main__":
    main()