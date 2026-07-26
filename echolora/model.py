import torch.nn as nn
from .layer import EchoLoraLinear


def _get_parent_and_child(model, target_name):
    """Given 'transformer.h.0.attn.c_attn', return (parent_module, 'c_attn')"""
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_echo_lora(model, config):
    """
    Wraps target linear layers in the model with EchoLoraLinear.
    Freezes all non-LoRA parameters.
    """
    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # replace target modules
    for name, module in model.named_modules():
        if config.target_modules is None:
            break
        for target in config.target_modules:
            if name.endswith(target):  # remove isinstance check
                parent, child_name = _get_parent_and_child(model, name)
                new_layer = EchoLoraLinear(
                    base_layer=module,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    bottleneck_dim=config.bottleneck_dim,
                )
                setattr(parent, child_name, new_layer)

    return model
