import os
import torch
from typing import Optional, List
from torch import Tensor

from evaluator.coco_evaluator import COCOAPIEvaluator
from dataset.coco import build_coco
from dataset.transforms import build_transform


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms'][args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = build_transform(
        is_train=True, 
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        min_size=cfg['train_min_size'],
        max_size=cfg['train_max_size'],
        random_size=cfg['random_size']
    )
    val_transform = build_transform(
        is_train=False, 
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        min_size=cfg['test_min_size'],
        max_size=cfg['test_max_size'],
        random_size=None
    )

    # dataset
    if args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        # dataset
        dataset = build_coco(
            root=args.root,
            transform=train_transform,
            return_masks=False
            )
        # evaluator
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=val_transform
            )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                        batch_size, 
                                                        drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler_train,
                                             collate_fn=collate_fn, 
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    
    return dataloader
    

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def load_weight(device, model, path_to_ckpt):
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
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
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    model = model.to(device).eval()
    print('Finished loading model!')

    return model


class CollateFunc(object):
    def _max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes


    def __call__(self, batch):
        batch = list(zip(*batch))

        image_list = batch[0]
        target_list = batch[1]

        # TODO make this more general
        if image_list[0].ndim == 3:

            # TODO make it support different-sized images
            max_size = self._max_by_axis([list(img.shape) for img in image_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in image_list]))
            batch_shape = [len(image_list)] + max_size
            b, c, h, w = batch_shape
            dtype = image_list[0].dtype
            device = image_list[0].device
            batch_tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            batch_mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

            for img, pad_img, m in zip(image_list, batch_tensor, batch_mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError('not supported')
            
        return batch_tensor, target_list, batch_mask

