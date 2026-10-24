# Classifier baselines

## Shared protocol
The five classifier baselines (EfficientNet-B0/B3, ConvNeXt-Tiny, ResNet-50, and Swin-Tiny) are presented under one common training and evaluation protocol. The published protocol uses TRAIN for parameter updates, DEV for checkpoint selection, and CAL for referral-threshold selection. External datasets are used for evaluation.

## Artifact mapping
The documented workflow comprises 15,001 base-training updates followed by 1,500 classifier-head adaptation and 4,500 joint-training updates. Each backbone has a designated base checkpoint, selected continuation checkpoint, and fixed referral policy. See native_baseline_lineage.csv for the five corresponding artifact records. The registered selected step is 21,001 for B0 and 18,001 for the other four backbones.

## Record scope
The public parent archive and all five parent checkpoint SHA-256 values were verified. Selected-model hashes and CAL-prediction hashes in the table were cross-checked between the repository registry and policy files; those files were not independently downloaded in this check. The five parent checkpoints do not embed data_protocol or data_contract_sha256. Historical training membership therefore requires linkage to the original training manifests and run records; the common protocol description alone does not establish that linkage.


Sources: [usage](USAGE.md), [model index](../checkpoints/models.csv), and [artifact mapping](native_baseline_lineage.csv).
