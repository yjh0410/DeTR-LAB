import torch
from .detr import DeTR
from .criterion import build_criterion


# build DeTR detector
def build_detr(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False,
                pretrained=None,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    model = DeTR(
        cfg=cfg,
        device=device,
        num_classes=num_classes,
        trainable=trainable,
        aux_loss=args.aux_loss,
        use_nms=args.use_nms
    )

    print('==============================')
    print('Model Configuration: \n', cfg)

    # Load pretrained weight
    if pretrained is not None:
        print('Loading pretrained weight: ', pretrained)
        checkpoint = torch.load(pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
                    print(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)
                        
    # keep training
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    # build criterion for training
    if trainable:
        criterion = build_criterion(
            cfg=cfg,
            num_classes=num_classes,
            aux_loss=args.aux_loss
            )
    else:
        criterion = None

    return model, criterion
