from .detr_config import detr_config
from .anchor_detr_config import anchor_detr_config

def build_config(args):
    
    if args.version in ['detr_r18', 'detr_r50', 'detr_r50-DC5',
                        'detr_r101', 'detr_r101-DC5']:
        cfg = detr_config[args.version]

    elif args.version in ['anchor_detr_r18', 'anchor_detr_r50', 'anchor_detr_r50-DC5',
                          'anchor_detr_r101', 'anchor_detr_r101-DC5']:
        cfg = anchor_detr_config[args.version]

    else:
        print('Unknown Model !')
        exit(0)

    return cfg
