from .detr.build import build_detr
from .anchor_detr.build import build_anchor_detr



# build detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False,
                pretrained=None,
                resume=None):    
    if args.version in ['detr_r18', 'detr_r50', 'detr_r50-DC5',
                        'detr_r101', 'detr_r101-DC5']:
        model, criterion = build_detr(
            args=args,
            cfg=cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            pretrained=pretrained,
            resume=resume
        )
    elif args.version in ['anchor_detr_r18', 'anchor_detr_r50', 'anchor_detr_r50-DC5',
                          'anchor_detr_r101', 'anchor_detr_r101-DC5']:
        model, criterion = build_anchor_detr(
            args=args,
            cfg=cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            pretrained=pretrained,
            resume=resume
        )

    return model, criterion
