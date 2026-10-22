"""Inference-only structures for the 57 registered selected checkpoints.

No training runtime, public initialization file, asset installer or parent checkpoint.
Strict state loading is mandatory. Public dependency paths are checked statically; see docs/USAGE.md.
"""
import importlib
import os
import sys
from pathlib import Path

import torch
from torch import nn
import timm
from unittest.mock import patch

VENDOR = Path(__file__).resolve().parent / 'main/vendor'
if os.name == 'nt':
    # Imported vendor module paths can exceed Windows MAX_PATH.
    VENDOR = Path(chr(92) * 2 + '?' + chr(92) + str(VENDOR))
ARCHIVED_SOURCE = VENDOR / 'assets/archive/model_sources/model_core/src'
S2_SOURCE = VENDOR / 'assets/GROUP_1/model_dependencies/S2/code/scripts/dr/sparrow'
FACTORIES = {'B3': 'efficientnet_b3', 'ResNet50': 'resnet50',
             'SwinTiny': 'swin_tiny_patch4_window7_224', 'ConvNeXtTiny': 'convnext_tiny.fb_in1k'}
CHANNELS = {'B0': (40, 112), 'B3': (48, 136), 'ResNet50': (1024, 2048),
            'SwinTiny': (384, 768), 'ConvNeXtTiny': (384, 768)}


def dependencies(backbone):
    root = ARCHIVED_SOURCE if backbone == 'B3' else S2_SOURCE
    if not root.is_dir():
        raise FileNotFoundError(f'Bundled model sources missing: {root}')
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(VENDOR / 'code'))
    prefix = 'tasks.diabetic_retinopathy' if backbone == 'B3' else 'clean_s2_vendor.tasks.diabetic_retinopathy'
    clinical = importlib.import_module(prefix + '.models.clinical_risk.crrl_model')
    dpm = importlib.import_module(prefix + '.models.distributional_progression.distributional_progression_model')
    axis = importlib.import_module(prefix + '.sparrow_dr.severity').OrderedSeverityAxis
    destyle = importlib.import_module(prefix + '.sparrow_dr.destyle').ResidualDestylizer
    gaussian = importlib.import_module('numerics').gaussian_logp
    return clinical.ClinicalRiskRepresentationLearningModel, dpm, axis, destyle, gaussian


class Pyramid(nn.Module):
    """Same stage boundaries and serialization names as the deployed main models."""
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == 'B0':
            from torchvision.models import efficientnet_b0
            net = efficientnet_b0(weights=None)
            self.backbone = nn.Sequential(*list(net.features.children())[:6])
        else:
            kw = {'img_size': 512} if name == 'SwinTiny' else {}
            self.backbone = timm.create_model(FACTORIES[name], pretrained=False, num_classes=1000, **kw)
            self.backbone.reset_classifier(0)

    def forward(self, x):
        b = self.backbone
        if self.name == 'B0':
            for i in range(4):
                x = b[i](x)
            return x, b[5](b[4](x))
        if self.name == 'ResNet50':
            x = b.layer2(b.layer1(b.maxpool(b.act1(b.bn1(b.conv1(x))))))
            s3 = b.layer3(x)
            return s3, b.layer4(s3)
        if self.name == 'SwinTiny':
            x = b.layers[1](b.layers[0](b.patch_embed(x)))
            s3 = b.layers[2](x)
            s4 = b.norm(b.layers[3](s3))
            return s3.permute(0, 3, 1, 2).contiguous(), s4.permute(0, 3, 1, 2).contiguous()
        if self.name == 'ConvNeXtTiny':
            x = b.stages[1](b.stages[0](b.stem(x)))
            s3 = b.stages[2](x)
            return s3, b.norm_pre(b.stages[3](s3))
        raise ValueError(self.name)


class MainModel(nn.Module):
    def __init__(self, backbone, *, variant="FULL", query_count=6):
        super().__init__()
        self.variant = variant
        clinical, dpm, axis, destyle, gaussian = dependencies(backbone)
        args = dict(backbone_name='efficientnet_b3', pretrained=False, backbone_img_size=512,
                    pooling='gem', local_query_count=query_count, feature_dim=256, attention_heads=4,
                    mode='local_global', beta=1., crrl_mode='consistency',
                    risk_hidden_dim=256, risk_embedding_dim=256)
        # B3 uses the original features-only factory (out_indices 2,3), as in P4.
        if backbone == 'B3':
            base = clinical(**args)
        else:
            pyramid = Pyramid(backbone)
            c3, c4 = CHANNELS[backbone]
            with patch.object(dpm, '_build_feature_pyramid', return_value=(pyramid, c3, c4)):
                base = clinical(**args)
        self.inner = nn.Module()
        self.inner.base = base
        c3, c4 = CHANNELS[backbone]
        self.inner.axis_stage3 = axis(c3)
        self.inner.axis_stage4 = axis(c4)
        self.inner.destylize_stage3 = destyle(c3)
        self.inner.destylize_stage4 = destyle(c4)
        self.gaussian_logp = gaussian
        # Mirror ablation/structural.py without importing its training runtime.
        if variant == 'NO_DESTYLE':
            self.inner.destylize_stage3 = nn.Identity()
            self.inner.destylize_stage4 = nn.Identity()
        elif variant == 'NO_LOCAL':
            for name in ('local_branch', 'local_project', 'local_head', 'fusion'):
                delattr(self.inner.base, name)
            for name in ('destylize_stage3', 'axis_stage3'):
                delattr(self.inner, name)

    def forward(self, x):
        base = self.inner.base
        s3, s4 = base.feature_pyramid(x)
        if self.variant == 'NO_LOCAL':
            zg = base.global_branch(self.inner.destylize_stage4(s4))
            mu, sigma = base.global_head(zg)
            return self.gaussian_logp(mu, sigma)
        local = base.local_branch(self.inner.destylize_stage3(s3))
        zl = base.local_project(local['local_embedding_raw'])
        zg = base.global_branch(self.inner.destylize_stage4(s4))
        lm, ls = base.local_head(zl)
        gm, gs = base.global_head(zg)
        fused = base.fusion(local_mu=lm, local_sigma=ls, global_mu=gm, global_sigma=gs)
        # The continuation uses stable FP64 Gaussian probabilities, not base._probs.
        return self.gaussian_logp(fused.mean, fused.std)


def build_model(family, backbone, *, variant='FULL', query_count=6):
    if backbone not in CHANNELS or family not in ('main', 'ablation', 'sensitivity', 'native'):
        raise ValueError('Unsupported family/backbone')
    if backbone == 'SwinTiny' and family == 'sensitivity':
        raise ValueError('SwinTiny sensitivity is not registered')
    if type(query_count) is not int or query_count not in (2, 4, 6, 8, 10):
        raise ValueError('Unsupported local query count')
    if family == 'ablation':
        if variant not in ('NO_LOCAL', 'NO_DESTYLE', 'NO_SAM') or query_count != 6:
            raise ValueError('Ablation requires one registered variant and the default K=6')
    elif variant != 'FULL' or (family != 'sensitivity' and query_count != 6):
        raise ValueError('Unexpected variant/query count for this family')
    if family != 'native':
        # NO_SAM and rho affect training; their inference graph remains FULL.
        return MainModel(backbone, variant=variant, query_count=query_count)
    if backbone == 'B0':
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
        return model
    kwargs = {'img_size': 512} if backbone == 'SwinTiny' else {}
    return timm.create_model(FACTORIES[backbone], pretrained=False, num_classes=5, **kwargs)
