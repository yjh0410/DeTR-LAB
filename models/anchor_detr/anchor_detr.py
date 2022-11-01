import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

from .backbone import build_backbone
from .transformer import build_transformer

import utils.box_ops as box_ops
from utils.nms import multiclass_nms


# AnchorDeTR detector
class AnchorDeTR(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=91,
                 trainable=False,
                 aux_loss=False,
                 use_nms=False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = cfg['conf_thresh']
        self.nms_thresh = cfg['nms_thresh']
        self.stride = cfg['stride']
        self.hidden_dim = cfg['hidden_dim']
        self.pretrained=cfg['pretrained'] and trainable

        self.aux_loss = aux_loss
        self.use_nms = use_nms
        

        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dims = build_backbone(cfg, self.pretrained, False)
        
        ## input proj layer
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bk_dims[-1], self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            )])

        ## transformer
        self.transformer = build_transformer(cfg)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: (Tensor) [Nq, C]
            box_pred: (Tensor) [Nq, 4]
        """
        # (HxWxAxK,)
        cls_pred = cls_pred.flatten().sigmoid_()

        # Keep top k top scoring indices only.
        num_topk = min(100, box_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        labels = topk_idxs % self.num_classes

        bboxes = box_pred[topk_box_idxs]

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        if self.use_nms:
            # nms
            scores, labels, bboxes = multiclass_nms(
                scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # backbone
        x = self.backbone(x)
        x = self.input_proj[0](x["0"]) # [B, C, H, W]
        # generate pos embed
        mask = torch.zeros([x.shape[0], *x.shape[-2:]], device=x.device, dtype=torch.bool) # [B, H, W]

        # transformer
        outputs_class, outputs_coord = self.transformer(x, mask)

        # we only compute the loss of last output from decoder
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        # batch_size = 1
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # [B, N, C] -> [N, C]
        cls_pred = out_logits[0]
        box_pred = box_ops.box_cxcywh_to_xyxy(out_bbox)[0]

        # post process
        bboxes, scores, labels = self.post_process(cls_pred, box_pred)

        return bboxes, scores, labels
                

    def forward(self, x, mask=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            x = self.backbone(x)
            x = self.input_proj[0](x["0"]) # [B, C, H, W]

            # generate pos embed
            fmp_size = x.shape[-2:]
            if mask is not None:
                # [B, H, W]
                mask = F.interpolate(mask[None].float(), size=fmp_size).bool()[0]
            else:
                mask = torch.zeros([x.shape[0], *x.shape[-2:]], device=x.device, dtype=torch.bool)

            # transformer
            outputs_class, outputs_coord = self.transformer(x, mask)

            # we only compute the loss of last output from decoder
            outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)

            return outputs
