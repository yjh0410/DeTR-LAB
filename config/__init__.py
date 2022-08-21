from .detr_config import detr_config


def build_config(args):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if args.version in ['detr_r50', 'detr_r50-DC5', 'detr_r101']:
        cfg = detr_config

    return cfg
