from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is not None:
        try:
            if cuda_module.is_available():
                cuda_module.manual_seed_all(seed)
        except Exception:
            pass

    npu_module = getattr(torch, "npu", None)
    if npu_module is not None:
        try:
            if npu_module.is_available():
                npu_module.manual_seed_all(seed)
        except Exception:
            pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
