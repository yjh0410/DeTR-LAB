from .detr.build import build_detr



# build detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False,
                pretrained=None,
                resume=None):    
    if args.version in ['detr_r18', 'detr_r50', 'detr_r50-DC5', 'detr_r101', 'detr_r101-DC5', 'detr_r50-RT']:
        model, criterion = build_detr(
            args=args,
            cfg=cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            pretrained=pretrained,
            resume=resume
        )

    return model, criterion
