from .detr_config import detr_config


def build_config(args):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if args.version == 'detr':
        cfg = detr_config

    return cfg
