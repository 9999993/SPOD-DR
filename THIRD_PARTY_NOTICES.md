# Third-party components

Project-authored code, documentation, and the SPOD-DR checkpoints distributed
with this repository's releases are provided under the MIT License; see LICENSE.
This notice identifies separately governed dependencies and source materials.

Dependencies are installed separately: PyTorch, torchvision, timm, NumPy, Pillow,
SciPy, scikit-learn, safetensors and the packages in requirements-training.txt.
Their own licenses apply; retain notices required by the versions distributed.

Public pretrained initialization sources and revisions are listed in
checkpoints/public_initialization.json. Code licensing does not automatically
relicense third-party pretrained weights, source images, or labels. Study membership tables and derived evaluation predictions accompany the code.
Original dataset image collections and full local image/cache manifests are obtained
separately under the applicable dataset terms. Any inherited third-party notices
in source files remain.

## Method-figure input photograph

The input photograph in `docs/assets/spod-dr/method-overview.svg` and its PNG
preview is adapted from the [DeepDRiD dataset](https://github.com/deepdrdoc/DeepDRiD)
provided by Ruhan Liu et al., under
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The adapted input photograph retains its original license and is not
relicensed under MIT. This attribution applies to the repository figure,
not to the different input illustration in the current manuscript Fig. 1.

## Current manuscript Figure 1

As confirmed by the author on 24 October 2026, the current manuscript uses
an author-created fundus illustration for the input example, rather than
the DeepDRiD photograph in the repository figure. The earlier statement extending the repository image
attribution to manuscript Fig. 1 was incorrect for this manuscript version.
The original dataset terms remain applicable to research dataset images.

## DDR reference labels in evaluation attachments
DDR reference labels included in `spod-dr-ci-inputs.zip` originate from the [OIA-DDR dataset](https://github.com/nkicsl/DDR-dataset) by Tao Li et al., [Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening](https://doi.org/10.1016/j.ins.2019.06.011) (2019). These labels retain the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/), including attribution, non-commercial use and share-alike conditions; they are not covered by this project's MIT license. Labels are included alongside model predictions for evaluation reproducibility.
