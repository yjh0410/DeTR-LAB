# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr

"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x)
        fmp_list = dict()
        for name, fmp in xs.items():
            fmp_list[name] = fmp

        return fmp_list


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, 
                 model_name: str,
                 pretrained: bool,
                 dilation: bool,
                 norm_type: str,
                 return_interm_layers: bool):
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, model_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        num_channels = [128, 256, 512] if model_name in ('resnet18', 'resnet34') else [512, 1024, 2048]
        super().__init__(backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone):
        super().__init__(backbone)

    def forward(self, x):
        return self[0](x)


# build backbone
def build_backbone(cfg, pretrained=False, return_interm_layers=False):
    # backbone
    backbone = Backbone(
        model_name=cfg['backbone'], 
        pretrained=pretrained, 
        dilation=cfg['res5_dilation'],
        norm_type=cfg['bk_norm'],
        return_interm_layers=return_interm_layers)
    bk_dims = backbone.num_channels

    model = Joiner(backbone)


    return model, bk_dims
    
    
if __name__ == '__main__':
    model, feat_dim = build_backbone(model_name='resnet18', pretrained=True, res5_dilation=False)
    print(feat_dim)

    x = torch.randn(2, 3, 800, 800)
    outputs = model(x)
    print(outputs["0"].shape)
    for k in outputs.keys():
        print(k)
        f = outputs[k]
        print(f.shape)
