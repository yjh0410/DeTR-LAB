from .resnet import build_resnet


def build_backbone(model_name='resnet18', pretrained=False, norm_type='BN', res5_dilation=False):
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        backbone, bk_dims = build_resnet(
            model_name=model_name,
            pretrained=pretrained,
            norm_type=norm_type,
            res5_dilation=res5_dilation
        )

    return backbone, bk_dims
    