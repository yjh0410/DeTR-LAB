# DeTR Configuration

detr_config = {
    'detr_r50': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'bk_norm': 'FrozeBN',
        'res5_dilation': False,
        'stride': 32,
        # transformer
        'hidden_dim': 256,
        'dropout': 0.1,
        'num_heads': 8,
        'mlp_dim': 2048,
        'num_encoders': 6,
        'num_decoders': 6,
        'pre_norm': False,
        'num_queries': 100,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # loss
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'eos_coef': 0.1,
        # training config
        'batch_size': 16,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'max_epoch': 150,
        'lr_epoch': [100],
        },

    'detr_r50-DC5': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'bk_norm': 'FrozeBN',
        'res5_dilation': True,
        'stride': 16,
        # transformer
        'hidden_dim': 256,
        'dropout': 0.1,
        'num_heads': 8,
        'mlp_dim': 2048,
        'num_encoders': 6,
        'num_decoders': 6,
        'pre_norm': False,
        'num_queries': 100,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # loss
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'eos_coef': 0.1,
        # training config
        'batch_size': 16,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'max_epoch': 150,
        'lr_epoch': [100],
        },

    'detr_r101': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # backbone
        'backbone': 'resnet101',
        'pretrained': True,
        'bk_norm': 'FrozeBN',
        'res5_dilation': False,
        'stride': 32,
        # transformer
        'hidden_dim': 256,
        'dropout': 0.1,
        'num_heads': 8,
        'mlp_dim': 2048,
        'num_encoders': 6,
        'num_decoders': 6,
        'pre_norm': False,
        'num_queries': 100,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'eos_coef': 0.1,
        # training config
        'batch_size': 16,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'max_epoch': 150,
        'lr_epoch': [100],
        },

    'detr_r101-DC5': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # backbone
        'backbone': 'resnet101',
        'pretrained': True,
        'bk_norm': 'FrozeBN',
        'res5_dilation': True,
        'stride': 16,
        # transformer
        'hidden_dim': 256,
        'dropout': 0.1,
        'num_heads': 8,
        'mlp_dim': 2048,
        'num_encoders': 6,
        'num_decoders': 6,
        'pre_norm': False,
        'num_queries': 100,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # loss
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'eos_coef': 0.1,
        # training config
        'batch_size': 16,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'max_epoch': 150,
        'lr_epoch': [100],
        },

}