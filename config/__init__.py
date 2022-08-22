from .detr_config import detr_config


def build_config(args):
    
    if args.version in ['detr_r18', 'detr_r50', 'detr_r50-DC5', 'detr_r101', 'detr_r101-DC5', 'detr_r50-RT']:
        cfg = detr_config[args.version]

    else:
        print('Unknown Model !')
        exit(0)

    return cfg
