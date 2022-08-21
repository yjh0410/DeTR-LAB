import torch
from .detr.detr import DeTR


# build DeTR detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False, 
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if args.version in ['detr_r50', 'detr_r50-DC5', 'detr_r101', 'detr_r101-DC5']:
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
    if resume is not None:
        print('Loading pretrained weight: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            print(k)
        exit(0)
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
                    print(k)
            else:
                print(k)

        model.load_state_dict(checkpoint_state_dict)
                        
    return model
