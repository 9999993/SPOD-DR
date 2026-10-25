# Third-party components

Project-authored code, documentation, and the SPOD-DR checkpoints distributed
with this repository's releases are provided under the MIT License; see LICENSE.

Dependencies are installed separately: PyTorch, torchvision, timm, NumPy, Pillow,
SciPy, scikit-learn, safetensors and the packages in requirements-training.txt.
Their own licenses apply; retain notices required by the versions distributed.

Public pretrained initialization sources and revisions are listed in
[checkpoints/public_initialization.json](checkpoints/public_initialization.json).
Third-party pretrained weights, images, and labels retain their original licenses.
Study membership tables and derived evaluation predictions accompany the code;
obtain original dataset images and full local image/cache manifests under the
applicable dataset terms. Retain inherited notices in source files.

## Method-figure illustration

The input illustration in `docs/assets/spod-dr/method-overview.svg` and its PNG
preview comes from the repository's cover illustration. The same figure appears
in manuscript Fig. 1.

## DDR reference labels in evaluation attachments
DDR reference labels included in `spod-dr-ci-inputs.zip` originate from the [OIA-DDR dataset](https://github.com/nkicsl/DDR-dataset) by Tao Li et al., [Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening](https://doi.org/10.1016/j.ins.2019.06.011) (2019). These labels accompany model predictions and retain the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/), including attribution, non-commercial use and share-alike conditions.
