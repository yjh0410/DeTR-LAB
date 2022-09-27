import torch


def build_lr_scheduler(cfg, optimizer=None, resume=None):
    print('==============================')
    print('Lr Scheduler: {}'.format(cfg['lr_scheduler']))

    if cfg['lr_scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=cfg['lr_epoch']
            )

    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("lr_scheduler")
        lr_scheduler.load_state_dict(checkpoint_state_dict)
                        
                                
    return optimizer
