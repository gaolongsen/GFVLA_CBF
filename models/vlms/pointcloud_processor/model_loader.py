import os
import pathlib
import sys

import torch

# from gfvla_env.helpers.pytorch import log_params_to_file
from .gfvla_env_dino import GfvlaEnvDinov2
from .model_utils.clip_loralib import (
    LoRALayer,
    apply_lora,
    merge_lora,
)
from .model_utils.mv_utils import cfg_from_yaml_file


def set_trainable_params(model, model_type="clip"):
    if model_type == "clip":
        substrings = ["cls_token", "cls_pos", "norm", "patch_embed", "patch_linear", "lora", "point_extractor", "pos_mlp"]
    elif model_type == "dinov2":
        substrings = ['linear_', 'cls_token', 'cls_pos', 'norm', 'patch_embed', 'qkv.bias']  
    for n, p in model.named_parameters():
        p.requires_grad = True
        if all(sub not in n for sub in substrings):
            p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad = True
    return model

def gfvla_env_dinov2_base(**kwargs):
    current_dir = pathlib.Path(__file__).parent
    yaml_path = os.path.join(current_dir, "model_config/ViT-B-14.yaml")
    config = cfg_from_yaml_file(yaml_path)
    model = GfvlaEnvDinov2(config=config.model, **kwargs)
    # set_trainable_params(model, model_type="dinov2")
    return model


