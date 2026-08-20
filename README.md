# HE-Aware Training

Fully homomorphic encryption (FHE) can run a language model without revealing the user's prompt, but the cost on GPT-2 is that generating one encrypted token takes about a minute. Most of that latency has two sources: ciphertexts carry a finite multiplicative depth budget that a bootstrap (a very slow and costly operation) must reset periodically. Even more, FHE schemes supports only additions and multiplications, thus nonlinearities must be replaced by iterative approximations, which trade latency for precision one iteration at a time, expanding the computational graph and increasing the total number of bootstrap needed to run a model. We introduce **Homomorphic-Encryption-Aware Training (HEAT)**, a fine-tuning method that makes the iteration count at every nonlinearity _learnable_, with no architectural changes and no retraining from scratch.

This repository contains the code necessary to replicate the depth reducing training
experiment. Encrypted inference runs on the [Perseus](https://github.com/gladia-research-group/perseus)
backend, vendored here as `src/perseus` and pinned to the `heat-baseline` tag.

## Layout

- `src/he_aware_training/`
  - `modules/` the learnable layers. `learnable_components.py` is the ponder machinery that makes depth differentiable, `mem_eff_ponder_functions.py` its memory-efficient autograd, `learnable_{thor,vit,bert}_layers.py` the per-approximation attention.
  - `approximation/classic.py` classical approximation machinery.
  - `scripts/` pipeline entry points, with `preprocess/` for calibration, data preparation and export
  - `utils/` data, surgery, loss, checkpointing, training loop.
- `configs/` Hydra configs. `he_aware_train{,_vit,_bert}.yaml` are the training entries; `model/approximation/` holds the per-arm circuit descriptions.
- `scripts/` scripts for FHE backend porting.
- `src/perseus/` the FHE backend (git submodule, tag `heat-baseline`). Reads the deploy
  config and the exported weights; not needed for training or calibration.

## Pipeline

Every stage is a Hydra entry point tuned by Hydra overrides. For training, the arm is
selected by `--config-name he_aware_train{,_vit,_bert}`, not by `model=`: the entry point
pins `config_name="he_aware_train"`, so `model=vit` alone would compose the ViT model
against GPT-2's dataset, trainer and regularizer. The other stages take `model=`. Below is
one full pass for GPT-2, with the ViT/BERT differences.

### 0. Install

```bash
git clone --recursive https://github.com/<org>/he-aware-training.git
cd he-aware-training
uv sync
source .venv/bin/activate
export PROJECT_ROOT=$PWD
```

`--recursive` is required: Perseus carries its own nested dependencies. On an existing
clone, `git submodule update --init --recursive` does the same. Training and
calibration do not need it — only the encrypted run does.

`PROJECT_ROOT` anchors the config-relative paths. The other locations default
under the working directory:

| variable | used for | default |
|---|---|---|
| `PROJECT_ROOT` | repo root for config-relative paths | `.` |
| `HF_HOME` | HuggingFace cache, and the calibrator's image pool | `<cwd>/.cache` |
| `DATA_PATH` | prepared dataset root | `<cwd>/data` |

Beyond those, `WANDB_ENTITY` and `WANDB_PROJECT` are picked up if you log runs to Weights & Biases, `SLURM_ACCOUNT` and `SLURM_PARTITION` if you submit through the Hydra submitit launcher rather than running locally.

### 1. Prepare the data

```bash
python src/he_aware_training/scripts/prepare_data.py model=gpt2 dataset=openwebtext
```

For the image arm instead, memmaps at the deploy resolution, then the pool the calibrator reads (native 224; it interpolates down at batch time):

```bash
python -m he_aware_training.scripts.preprocess.prepare_eurosat_vit80 --data-path "$DATA_PATH"
python -m he_aware_training.scripts.preprocess.build_eurosat_calib_pool --n-images 512
```

### 2. Calibrate

Fits the approximation domains and the starting iteration counts, writing the
`configs.json` the trainer reads:

```bash
python -m he_aware_training.scripts.preprocess.calibrate \
    model=gpt2 \
    calib_out_path=configs/model/approximation/gpt2_base/configs.json
```

### 3. Train

Depth is learned here, under the ponder and regularizer terms:

```bash
python src/he_aware_training/scripts/train_he_aware_llm.py \
    --config-name he_aware_train \
    model.surgery.norm=true model.surgery.attention=true
```

We do not train GeLU as it's later approximated as a polynomial.

### 4. Recalibrate LayerNorm on the trained checkpoint

Training changes the activation and so their statistics, so the domains fitted in step 2 can be unstable without the clamping guards. Recalibrate the init constants against the trained checkpoint, then install them.

```bash
python -m he_aware_training.scripts.preprocess.calibrate \
    model=gpt2 trainer.init_from=checkpoints/gpt2/<run-id>/last.pt \
    calib_out_path=checkpoints/gpt2/<run-id>/recalibrated/configs.json

python scripts/splice_ln_recalib.py \
    --base       configs/model/approximation/gpt2_base/configs.json \
    --calib      checkpoints/gpt2/<run-id>/recalibrated/configs.json \
    --ckpt       checkpoints/gpt2/<run-id>/last.pt \
    --out-config configs/model/approximation/gpt2_heat \
    --out-ckpt   checkpoints/gpt2/heat_lnr
```

### 5. Build the deploy config

Reads the learned counts out of the checkpoint and writes them into the config, reporting the resulting circuit depth:

```bash
python scripts/make_circuit_config.py --arch gpt2 \
    --ckpt   checkpoints/gpt2/heat_lnr/last.pt \
    --src    configs/model/approximation/gpt2_heat/configs.json \
    --dst    src/perseus/configs/model/approximation/gpt2_heat \
    --mirror configs/model/approximation/gpt2_heat
```

Pass `--calib <dir> --counts-src <dir>` for `--arch vit`, `--calib <dir>` for `--arch bert`. `--src` must be the config carrying the step-4 domains: they exist only in a config, never in a checkpoint, so nothing here can recover them.

### 6. Plaintext evaluation

`eval_lm_benchmarks.py` scores the plaintext baseline and the approximated model side
by side on the lm-eval tasks (`eval.run_baseline` and `eval.run_hybrid`, both on by
default), so one run gives the comparison:

```bash
python src/he_aware_training/scripts/eval_lm_benchmarks.py model=gpt2 \
    eval.backbone_ckpt=checkpoints/gpt2/heat_lnr/last.pt

python -m he_aware_training.scripts.eval_vit_heat_acc \
    --ckpt checkpoints/vit-base-patch16-224/<run-id>/last.pt \
    --calib configs/model/approximation/vit_squeeze
```

This gates a candidate rather than settling it: encrypted execution can invert
simulated rankings, so step 8 is the arbiter.

### 7. Export

The two artifacts Perseus consumes, alongside the step-5 config:

```bash
python -m he_aware_training.scripts.preprocess.save_weights \
    model=gpt2 eval.backbone_ckpt=checkpoints/gpt2/heat_lnr/last.pt

python -m he_aware_training.scripts.preprocess.gather_all_blocks_test_data \
    model=gpt2 +steps_T=16
```

The oracle rows are only needed to check an encrypted run against plaintext; free-running generation does not require them.

Perseus reads the deploy config from its own tree, its loader takes a directory and parses `configs.json` out of it, which is why step 5 writes there and `--mirror` keeps the training tree's copy identical.

### 8. Run encrypted

Building the backend needs a GPU node and its own dependency chain; login nodes will not do:

```bash
cd src/perseus
bash  scripts/install_deps.sh                        # OpenFHE + FIDESlib
cmake --build build-py --parallel 16 --target _core  # or pass BUILD=1 to run_task.sh
```

`scripts/run_task.sh` is the measured path, one task per job, discriminated entirely by environment. `STAGE=run`, executes a **planned** graph; getting there is capture → plan → run:

```bash
TASK=decode STAGE=capture sbatch scripts/run_task.sh   # graph capture
bash scripts/make_plans.sh                             # login-side plan pass
TASK=decode sbatch scripts/run_task.sh                 # planned run (STAGE=run default)
```

| variable | values |
|---|---|
| `TASK` | `decode` `gen` `vit80` `vit112` |
| `STAGE` | `run` planned (default) · `eager` unplanned · `capture` graph capture |
| `RUNNER` | `python` the in-process Perseus module (default) · `cuda` the native CLI, GPT-2 only |
| `BUILD=1` | build `_core` as part of the job |

You can also run in `eager` mode, without a fixed bootstrap schedule. Such a run is correct but likely materially slower. We ship pre-computed plans to reproduce our baselines.

Submit from a clean shell with no modules loaded, and see `src/perseus/README.md` for the rest of the backend documentation.
