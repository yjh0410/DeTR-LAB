import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from typing import List

from evaluator.coco_evaluator import COCOAPIEvaluator
from dataset.coco import build_coco
from dataset.transforms import build_transform


def build_dataset(cfg, args, device):
    # transform
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
        num_classes = 91
        # dataset
        dataset = build_coco(
            root=data_dir,
            transform=train_transform,
            is_train=True,
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
        sampler = DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    return dataloader
    

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


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


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def load_weight(model, path_to_ckpt):
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

    print('Finished loading model!')

    return model


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

