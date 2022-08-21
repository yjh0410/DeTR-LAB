import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

from ...backbone import build_backbone
from .transformer import build_transformer
from .mlp import MLP

from .loss import build_criterion

import utils.box_ops as box_ops


# DeTR detector
class DeTR(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=80,
                 trainable=False,
                 aux_loss=False,
                 use_nms=False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.num_queries = cfg['num_queries']
        self.conf_thresh = cfg['conf_thresh']
        self.nms_thresh = cfg['nms_thresh']
        self.stride = cfg['stride']
        self.hidden_dim = cfg['hidden_dim']

        self.aux_loss = aux_loss
        self.use_nms = use_nms

        # --------- Object Query ----------
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dims = build_backbone(
            model_name=cfg['backbone'],
            pretrained=cfg['pretrained'],
            norm_type=cfg['bk_norm'],
            res5_dilation=cfg['res5_dilation']
            )
        
        ## input proj layer
        self.input_proj = nn.Conv2d(bk_dims[-1], self.hidden_dim, kernel_size=1)

        ## transformer
        self.transformer = build_transformer(cfg)

        ## output
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

        # criterion
        if self.trainable:
            self.criterion = build_criterion(
                cfg=cfg,
                device=device,
                num_classes=num_classes,
                aux_loss=aux_loss
            )


    # Position Embedding
    def position_embedding(self, mask, fmp_size, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        hs, ws = fmp_size
        
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert mask is not None
        not_mask = ~mask
        # [B, H, W]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (hs + eps) * scale
            x_embed = x_embed / (ws + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # [B, d, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos


    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    @torch.no_grad()
    def inference(self, x):
        # backbone
        x = self.backbone(x)
        x = self.input_proj(x['layer4'])

        # generate pos embed
        mask = torch.ones([x.shape[0], *x.shape[-2:]], device=x.device, dtype=torch.bool) # [B, H, W]
        fmp_size = x.shape[-2:]
        pos_embed = self.position_embedding(mask, fmp_size)

        # transformer
        h = self.transformer(x, self.query_embed.weight, pos_embed, mask)[0]

        # output: [M, B, N, C] where M = num_decoder since we use all intermediate outputs of decoder
        outputs_class = self.class_embed(h)
        outputs_coord = self.bbox_embed(h).sigmoid()

        # we only compute the loss of last output from decoder
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)
        
        # batch_size = 1
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # [B, N, C] -> [N, C]
        prob = out_logits[0].softmax(-1)
        scores, labels = prob[..., :-1].max(-1)

        # xywh -> xyxy
        bboxes = box_ops.box_cxcywh_to_xyxy(out_bbox)[0]

        # intermediate outputs
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # batch_size = 1
                out_logits_i, out_bbox_i = aux_outputs['pred_logits'], aux_outputs['pred_boxes']
                # [B, N, C] -> [N, C]
                prob_i = out_logits_i[0].softmax(-1)
                scores_i, labels_i = prob_i[..., :-1].max(-1)

                # xywh -> xyxy
                bboxes_i = box_ops.box_cxcywh_to_xyxy(out_bbox_i)[0]

                scores = torch.cat([scores, scores_i], dim=0)
                labels = torch.cat([labels, labels_i], dim=0)
                bboxes = torch.cat([bboxes, bboxes_i], dim=0)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        if self.use_nms:
            # nms
            keep = np.zeros(len(bboxes), dtype=np.int)
            for i in range(self.num_classes):
                inds = np.where(labels == i)[0]
                if len(inds) == 0:
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_keep = self.nms(c_bboxes, c_scores)
                keep[inds[c_keep]] = 1

            keep = np.where(keep > 0)
            scores = scores[keep]
            labels = labels[keep]
            bboxes = bboxes[keep]

        return bboxes, scores, labels
                

    def forward(self, x, mask=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            x = self.backbone(x)
            x = self.input_proj(x['layer4'])

            # generate pos embed
            fmp_size = x.shape[-2:]
            if mask is not None:
                # [B, H, W]
                mask = F.interpolate(mask[None], size=fmp_size).bool()
            else:
                mask = torch.ones([x.shape[0], *x.shape[-2:]], device=x.device, dtype=torch.bool)

            pos_embed = self.position_embedding(mask, fmp_size)

            # transformer
            h = self.transformer(x, self.query_embed.weight, pos_embed, mask)[0]

            # output: [M, B, N, C] where M = num_decoder since we use all intermediate outputs of decoder
            outputs_class = self.class_embed(h)
            outputs_coord = self.bbox_embed(h).sigmoid()

            # we only compute the loss of last output from decoder
            outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)
            
            loss_dict = self.criterion(outputs)

            return loss_dict
