# Usage

Download and extract the [v1.0.0 source code (ZIP)](https://github.com/9999993/SPOD-DR/archive/refs/tags/v1.0.0.zip) before running the commands below.

Run commands from the repository root. See [model downloads](../README.md#model-downloads) for weights and [results](RESULTS.md) for figures and statistical definitions.

[Inference](#inference) · [Data preparation](#data-preparation) · [Training](#training) · [Confidence intervals](#confidence-intervals)

## Inference

Use Python 3.11 and a compatible PyTorch/torchvision pair for your hardware:

```bash
python -m pip install -r requirements.txt
```

### Input manifest

Provide a nonempty JSON array:

```json
[
  {
    "image_id": "example-image",
    "cache_path": "cache/example.npy",
    "cache_sha256": "<64-character SHA256 of the cache file>"
  }
]
```

Image IDs must be unique. Relative cache paths resolve against the manifest directory. Caches contain finite float32 values in CHW shape `(3, 512, 512)`, with ImageNet normalization. Replace the example digest with the cache file's SHA256.

### Preprocessing

Create a CSV with `image_path,cache_path` columns. Use a new cache directory and an existing directory for the report:

```bash
python data/preprocessing/generic/scripts/build_diabetic_retinopathy_fundus_cache.py --index-files local-data/image_cache_map.csv --input-size 512 --normalization imagenet --num-workers 1 --report-path local-data/cache_report.json
```

The pipeline converts to RGB, takes an integer-centered square crop, resizes with bicubic interpolation, divides by 255, applies ImageNet mean/std normalization, and converts to CHW. Existing caches are reused only when their stored dtype is float32, their shape is `(3, input_size, input_size)`, and all values are finite; other caches are rebuilt from the source images. Use a new directory when changing preprocessing. The DeepDRiD-specific builder is `data/preprocessing/deepdrid/build_deepdrid_512_cache.py`.

### Model selection and prediction

Backbone identifiers are `B0`, `B3`, `ConvNeXtTiny`, `ResNet50`, and `SwinTiny`. Experiment settings are in `configs/<family>/`; inference selects weights and their fixed POLICY through [models.csv](../checkpoints/models.csv).

| Family | Additional options | Coverage |
|---|---|---|
| `main` | None; default K=6, rho=0.025 | Five backbones |
| `ablation` | `--variant NO_LOCAL`, `NO_DESTYLE`, or `NO_SAM` | Five backbones |
| `sensitivity` | `--queries` 2, 4, 8, or 10; or `--rho` 0.00625, 0.0125, 0.05, or 0.1 | Five backbones |
| `native` | None | Five standard classifier baselines |

Choose a new output directory for each invocation. CPU is the default; add `--device cuda --batch-size 1` for GPU inference.

```bash
python src/scripts/release_predict.py --family main --backbone SwinTiny --inputs local-data/inputs.json --output runs/main-predictions
python src/scripts/release_predict.py --family ablation --backbone SwinTiny --variant NO_LOCAL --inputs local-data/inputs.json --output runs/ablation-predictions
python src/scripts/release_predict.py --family sensitivity --backbone B3 --queries 4 --inputs local-data/inputs.json --output runs/k4-predictions
python src/scripts/release_predict.py --family sensitivity --backbone ResNet50 --rho 0.05 --inputs local-data/inputs.json --output runs/rho-predictions
python src/scripts/release_predict.py --family native --backbone ResNet50 --inputs local-data/inputs.json --output runs/native-predictions
```

`predictions.csv` contains predicted grades, class probabilities, log probabilities, referral scores, and decisions. Referral scores are float64 log odds of grades 2–4 against grades 0–1; decisions use the fixed POLICY threshold with `>=`. `receipt.json` records the model and inputs for the local run. Store inputs in `local-data/` and outputs in `runs/`.

## Data preparation

All 65 models share the CSV membership tables in [data/manifests/common/](../data/manifests/common/). [model_splits.csv](../data/manifests/model_splits.csv) maps artifact IDs to these tables, with paths relative to `data/manifests/`.

| Role | Images | Source or purpose |
|---|---:|---|
| TRAIN | 23,987 | EyePACS 20,242; APTOS 2,788; DeepDRiD Set A 957 |
| DEV | 2,984 | EyePACS 2,524; APTOS 344; DeepDRiD Set A 116; checkpoint selection |
| CAL | 2,524 | EyePACS; threshold fitting |
| DDR | 12,422 | Evaluation |
| IDRiD | 103 | Evaluation |
| MESSIDOR-2 | 1,744 | Evaluation |

Evaluation images are excluded from TRAIN, DEV, and CAL. Obtain dataset images and labels under their respective terms, then create local caches and JSON manifests. The CSVs supply membership and row order. Join by `(source, image_id)`, preserving `row_index`, filename suffixes, row order, and the supplied DEV folds. Normalize source aliases `ddr_external` → DDR and `idrid_external_test` → IDRiD consistently when joining. Evaluation labels for CI are included in its prediction attachment.

[Cleaning counts](source_cleaning_counts.csv)

### Training manifests

Each role uses a JSON array with these fields:

| Field | Value |
|---|---|
| `image_id` | Nonempty string, unique within the role |
| `source` | Dataset name, consistent across roles |
| `grade` | Integer 0–4, or its integer-string representation |
| `cache_path` | Preprocessed NPY cache path |
| `cache_sha256` | Cache file's 64-character SHA256 |
| `component_hash` | Stable group identifier shared by duplicate or related images |

If present, `label` must equal `int(grade >= 2)`. Preserve known string `patient_id` and `exam_id` fields and hexadecimal `image_sha256` values. Cross-role validation uses these supplied identities. DEV and CAL must each contain both referral classes.

`DEV_FOLDS.csv` assigns every DEV image exactly once to fold 0–4 using `image_id,fold` columns. Each fold and its complement contain both referral classes; related components, patients, and examinations stay in one fold. The alternative `row_index,component_hash,fold` schema requires matching unique row indices and component IDs in the DEV manifest.

### Data configuration

Copy [dataset.example.json](../configs/dataset.example.json) into `local-configs/` and edit the paths. Required roles are TRAIN, DEV, and CAL. Declare evaluation cohorts when creating the run; remove the example HOLDOUT if unused. Relative manifest paths resolve against the configuration file, and cache paths against each manifest. Optional `expected_counts` and `expected_sha256` fields check supplied sizes and hashes.

Use DDR, IDRiD, and MESSIDOR-2 for the study evaluation roles. Custom evaluation names start with a letter, contain letters, digits, underscores or hyphens, are unique ignoring case, and cannot start with TRAIN, DEV, or CAL or use Windows reserved names.

Alternatively, pass `--inputs /path/to/manifests` with TRAIN.json, DEV.json, CAL.json, and DEV_FOLDS.csv. DDR.json, IDRiD.json, and MESSIDOR-2.json are discovered as evaluation cohorts; other names require `--data-config`.

## Training

Training and post-training evaluation use Linux, Python 3.11, and a compatible PyTorch/torchvision pair:

```bash
python -m pip install -r requirements-training.txt
```

Sensitivity training starts from public initialization and completes all 21,001 updates; ρ is used during the final 4,500 joint updates. The reported ρ experiments reused the common state at step 16,501.

### Public initialization

Main, ablation, sensitivity, and classifier base training use the following fixed initializations under `checkpoints/initialization/`:

| Backbone | Relative path | Fixed download |
|---|---|---|
| B0 | `B0/imagenet.pth` | [PyTorch](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth) |
| B3 | `B3/model.safetensors` | [Hugging Face](https://huggingface.co/timm/efficientnet_b3.ra2_in1k/resolve/0366a75518620e0f2077789202073759f2d61393/model.safetensors) |
| ConvNeXtTiny | `ConvNeXtTiny/model.safetensors` | [Hugging Face](https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/model.safetensors) |
| ResNet50 | `ResNet50/model.safetensors` | [Hugging Face](https://huggingface.co/timm/resnet50.a1_in1k/resolve/767268603ca0cb0bfe326fa87277f19c419566ef/model.safetensors) |
| SwinTiny | `SwinTiny/model.safetensors` | [Hugging Face](https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k/resolve/4cc3a7275b50b53a7bec45f32c236ebe64227cff/model.safetensors) |

Versions and file identities are in [public_initialization.json](../checkpoints/public_initialization.json). Override the default directory with `--initialization-dir local-assets/pretrained`, pointing to the directory containing the backbone subdirectories. Explicit relative initialization and output paths resolve against the working directory. Use the same initialization location for train and evaluate.

### Prepare, train, and evaluate

Create caches first, then prepare manifests and start training:

```bash
python src/scripts/training.py prepare --data-config local-configs/dataset.json --output runs/prepared
python src/scripts/training.py train --data-config local-configs/dataset.json --runs runs/experiments --family main --backbone B0
python src/scripts/training.py evaluate --data-config local-configs/dataset.json --runs runs/experiments --family main --backbone B0
```

`prepare` validates and assembles inputs; add `--verify-cache` to inspect cache contents. Training also performs preparation. `evaluate` uses a completed run: DEV selects a checkpoint, CAL fits its referral threshold, and evaluation cohorts supply final metrics. For a resource check, use `python src/scripts/training.py check --family main --backbone B0` with the same family, backbone, and initialization options. `prepare` and `check` use only the Python standard library.

The model-selection options are the same as in [inference](#model-selection-and-prediction). Continue a run in its existing directory with the same data, configuration, initialization, and code; use a new directory when changing them.

### Classifier baseline training

All five backbones support base training from public initialization followed by native continuation. The [base configuration](../configs/native_base.json) uses 15,001 updates, batch size 16, seed 2101, and AdamW at `3e-5`. All classifier parameters are trained on TRAIN. B0 and ConvNeXt-Tiny use FP64 log-softmax NLL; B3, ResNet-50, and Swin-Tiny use FP32 cross-entropy.

```bash
python src/scripts/training.py check --family native_base --backbone B0
python src/scripts/training.py train --data-config local-configs/dataset.json --runs runs/experiments --family native_base --backbone B0
```

This produces `runs/experiments/native_base/B0/checkpoints/B0/stage_15001.pt`. Continue from that parent and evaluate with the same data configuration:

```bash
python src/scripts/training.py train --data-config local-configs/dataset.json --runs runs/experiments --family native --backbone B0 --parent-checkpoint runs/experiments/native_base/B0/checkpoints/B0/stage_15001.pt
python src/scripts/training.py evaluate --data-config local-configs/dataset.json --runs runs/experiments --family native --backbone B0 --parent-checkpoint runs/experiments/native_base/B0/checkpoints/B0/stage_15001.pt
```

Replace `B0` with `B3`, `ConvNeXtTiny`, `ResNet50`, or `SwinTiny` throughout to train another backbone. To resume base training, repeat its command with the same inputs and run directory.

See the [shared baseline protocol and artifact mapping](NATIVE_BASELINES.md).

### Native continuation

Extract the `checkpoints/` directory from `models-baseline-parents.zip` into the repository root. Each `checkpoints/native_parents/<backbone>/stage_15001.pt` initializes a 6,000-step continuation:

```bash
python src/scripts/training.py train --data-config local-configs/dataset.json --runs runs/experiments --family native --backbone B0
```

Default parent paths are set by `parent.path` in [configs/native/](../configs/native/), relative to the repository root. Use `--parent-checkpoint` for a custom 15,001-step parent, and repeat the same option for evaluation. Base training is implemented in `src/scripts/native_base/worker.py`; continuation is in `src/scripts/native/`.

During joint optimization, B0 adapts `features[4:]` plus the classifier; B3 adapts `blocks[3:]`, `conv_head`, `bn2`, `global_pool`, and the classifier. The configurations list exact trainable parameters. The selected baseline checkpoint is at step 21,001 for B0 and step 18,001 for the other four backbones.

## Confidence intervals

Use saved predictions for five SPOD-DR models, five classifier baselines, three Swin-Tiny ablations, and ORDER-DR. Extract `local-data/ci/` from [ci-inputs.zip](https://github.com/9999993/SPOD-DR/releases/download/v1.0.0/ci-inputs.zip) into the repository root: 39 CSVs and five ORDER-DR NPZs with model metadata. [INPUT_REGISTRY.json](../reproduce/ci/INPUT_REGISTRY.json) maps models to inputs.

Use Python 3.12 for the statistical environment:

```bash
python -m pip install -r reproduce/ci/requirements.txt
python reproduce/ci/run_ci.py --inputs-root local-data/ci --policies-root . --profile all --check-inputs
python reproduce/ci/run_ci.py --inputs-root local-data/ci --policies-root . --profile all --run --output runs/ci
```

`main` covers same-backbone comparisons and SPOD-DR B0 minus ORDER-DR; `ablation` covers the three Swin-Tiny ablations minus FULL; `all` includes both. `--check-inputs` validates inputs; `--run` calculates intervals in a new output directory. See [statistical definitions](RESULTS.md#confidence-intervals).

Outputs are ABSOLUTE_CI.csv, PAIRED_CI.csv, cohort bootstrap arrays, and RUN.json. Rates and rate differences are proportions; multiply rate differences by 100 for percentage points. CI unit tests are in `reproduce/ci/test_ci_entrypoints.py`; model and training tests are under `tests/`.

## Grading outputs and referral decisions

Tables S15a and S15b cross-tabulate the saved predicted grades and referral decisions. After extracting `ci-inputs.zip` as above, run:

```bash
python reproduce/grade_referral_tables.py --inputs-root local-data/ci --output runs/grade-referral
```

The script uses the Python standard library, verifies prediction and POLICY hashes, and compares the stored FP64 log-odds with the fixed CAL thresholds. It exports [S15a.csv](assets/spod-dr/experiment-data/S15a.csv), [S15b.csv](assets/spod-dr/experiment-data/S15b.csv), and their input hashes. No model inference or threshold fitting is required. S15a reports referral-negative and referral-positive counts within each predicted grade. S15b gives reference grades among referral-positive images predicted as grade 0 or 1 by EfficientNet-B3 on MESSIDOR-2; referral positivity alone does not establish a false positive.
