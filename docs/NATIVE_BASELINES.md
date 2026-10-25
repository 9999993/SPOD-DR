# Classifier baselines

## Shared protocol
The five classifier baselines (EfficientNet-B0/B3, ConvNeXt-Tiny, ResNet-50, and Swin-Tiny) use a common training and evaluation protocol. The protocol uses TRAIN for parameter updates, DEV for checkpoint selection, and CAL for referral-threshold selection. External datasets are used for evaluation.

## Artifact mapping
Training comprises 15,001 base-training updates followed by 1,500 classifier-head adaptation and 4,500 joint-training updates. Each backbone has a designated base checkpoint, selected continuation checkpoint, and fixed referral policy. See native_baseline_lineage.csv for the five corresponding artifact records. The selected step is 21,001 for B0 and 18,001 for the other four backbones.

Sources: [usage](USAGE.md), [model index](../checkpoints/models.csv), and [artifact mapping](native_baseline_lineage.csv).
