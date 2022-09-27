# DeTR Configuration

anchor_detr_config = {
    'anchor_detr_r18': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # backbone
        'backbone': 'resnet18',
        'pretrained': True,
        'bk_norm': 'FrozeBN',
        'res5_dilation': False,
        'stride': 32,
        # transformer
        'hidden_dim': 256,
        'dropout': 0.1,
        'num_heads': 8,
        'mlp_dim': 1024,
        'num_encoders': 6,
        'num_decoders': 6,
        'num_query_position': 300,
        'num_query_pattern': 3,
        'spatial_prior': 'learned', # ['learned', 'grid']
        'attention_type': "RCDA",   # ['RCDA', 'nn.MultiheadAttention']
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # matcher
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,        
        # loss
        'loss_ce_coef': 2.0,
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'focal_alpha': 0.25,
        # training config
        'batch_size': 2,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # warmup
        'warmup': 'linear',
        'wp_iter': 500,
        'warmup_factor': 0.00066667,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        # lr scheduler
        'max_epoch': 50,
        'lr_epoch': [40],
        'lr_scheduler': 'step',
        },

    'anchor_detr_r50': {
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
        'mlp_dim': 1024,
        'num_encoders': 6,
        'num_decoders': 6,
        'num_query_position': 300,
        'num_query_pattern': 3,
        'spatial_prior': 'learned', # ['learned', 'grid']
        'attention_type': "RCDA",   # ['RCDA', 'nn.MultiheadAttention']
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # matcher
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,        
        # loss
        'loss_ce_coef': 2.0,
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'focal_alpha': 0.25,
        # training config
        'batch_size': 2,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # warmup
        'warmup': 'linear',
        'wp_iter': 500,
        'warmup_factor': 0.00066667,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        # lr scheduler
        'max_epoch': 50,
        'lr_epoch': [40],
        'lr_scheduler': 'step',
        },

    'anchor_detr_r50-DC5': {
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
        'mlp_dim': 1024,
        'num_encoders': 6,
        'num_decoders': 6,
        'num_query_position': 300,
        'num_query_pattern': 3,
        'spatial_prior': 'learned', # ['learned', 'grid']
        'attention_type': "RCDA",   # ['RCDA', 'nn.MultiheadAttention']
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # matcher
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,        
        # loss
        'loss_ce_coef': 2.0,
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'focal_alpha': 0.25,
        # training config
        'batch_size': 2,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # warmup
        'warmup': 'linear',
        'wp_iter': 500,
        'warmup_factor': 0.00066667,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        # lr scheduler
        'max_epoch': 50,
        'lr_epoch': [40],
        'lr_scheduler': 'step',
        },

    'anchor_detr_r101': {
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
        'mlp_dim': 1024,
        'num_encoders': 6,
        'num_decoders': 6,
        'num_query_position': 300,
        'num_query_pattern': 3,
        'spatial_prior': 'learned', # ['learned', 'grid']
        'attention_type': "RCDA",   # ['RCDA', 'nn.MultiheadAttention']
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.5,
        # matcher
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,        
        # loss
        'loss_ce_coef': 2.0,
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'focal_alpha': 0.25,
        # training config
        'batch_size': 2,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # warmup
        'warmup': 'linear',
        'wp_iter': 500,
        'warmup_factor': 0.00066667,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        # lr scheduler
        'max_epoch': 50,
        'lr_epoch': [40],
        'lr_scheduler': 'step',
        },

    'anchor_detr_r101-DC5': {
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
        'mlp_dim': 1024,
        'num_encoders': 6,
        'num_decoders': 6,
        'num_query_position': 300,
        'num_query_pattern': 3,
        'spatial_prior': 'learned', # ['learned', 'grid']
        'attention_type': "RCDA",   # ['RCDA', 'nn.MultiheadAttention']
        # post process
        'conf_thresh': 0.005,
        'nms_thresh': 0.5,
        # matcher
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,        
        # loss
        'loss_ce_coef': 2.0,
        'loss_bbox_coef': 5.0,
        'loss_giou_coef': 2.0,
        'focal_alpha': 0.25,
        # training config
        'batch_size': 2,
        'base_lr': 0.0001 / 16.,
        'bk_lr_ratio': 0.1,
        # warmup
        'warmup': 'linear',
        'wp_iter': 500,
        'warmup_factor': 0.00066667,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        # lr scheduler
        'max_epoch': 50,
        'lr_epoch': [40],
        'lr_scheduler': 'step',
        },

}