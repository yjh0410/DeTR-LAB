# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import inverse_sigmoid


from .row_column_decoupled_attention import MultiheadRCDA



class Transformer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_classes=91,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_query_position=300,
                 num_query_pattern=3,
                 spatial_prior="learned",
                 attention_type="RCDA"):
        super().__init__()
        # basic
        self.d_model = d_model
        self.nhead = nhead
        self.num_pattern = num_query_pattern
        self.attention_type = attention_type
        self.num_encoder_layers = num_encoder_layers
        self.spatial_prior = spatial_prior
        self.num_layers = num_decoder_layers
        self.num_classes = num_classes

        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead, attention_type)
        self.encoder_layers = _get_clones(encoder_layer, self.num_encoder_layers)

        # decoder
        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, nhead, attention_type)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)

        # pattern embed
        self.pattern = nn.Embedding(self.num_pattern, d_model)

        # pos embed
        self.num_position = num_query_position
        if self.spatial_prior == "learned":
            self.position = nn.Embedding(self.num_position, 2)

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )


        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self._reset_parameters()


    def _reset_parameters(self):

        num_pred = self.num_layers
        num_classes = 91
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.spatial_prior == "learned":
            nn.init.uniform_(self.position.weight.data, 0, 1)

        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])


    def forward(self, srcs, masks):

        # prepare input for decoder
        bs, c, h, w = srcs.shape

        # Position 
        if self.spatial_prior == "learned":
            reference_points = self.position.weight.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        elif self.spatial_prior == "grid":
            nx=ny=round(math.sqrt(self.num_position))
            self.num_position=nx*ny
            x = (torch.arange(nx) + 0.5) / nx
            y = (torch.arange(ny) + 0.5) / ny
            xy=torch.meshgrid(x,y)
            reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
            reference_points = reference_points.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        else:
            raise ValueError(f'unknown {self.spatial_prior} spatial prior')

        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(bs, 1, self.num_position, 1).reshape(
            bs, self.num_pattern * self.num_position, c)

        # Position embedding
        mask = masks
        pos_col, pos_row = mask2pos(mask)
        if self.attention_type=="RCDA":
            # decoupled row pos & col pos
            posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
            posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
            posemb_2d = None
        else:
            # 2d pos
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1), pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)],dim=-1)
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d))
            posemb_row = posemb_col = None

        outputs = srcs

        # Encoder layers
        for idx in range(len(self.encoder_layers)):
            outputs = self.encoder_layers[idx](outputs, mask, posemb_row, posemb_col,posemb_2d)

        srcs = outputs
        output = tgt

        outputs_classes = []
        outputs_coords = []
        # Decoder layers
        for lid, layer in enumerate(self.decoder_layers):
            output = layer(output, reference_points, srcs, mask, adapt_pos2d=self.adapt_pos2d,
                           adapt_pos1d=self.adapt_pos1d, posemb_row=posemb_row, posemb_col=posemb_col,posemb_2d=posemb_2d)
            reference = inverse_sigmoid(reference_points)
            outputs_class = self.class_embed[lid](output)
            tmp = self.bbox_embed[lid](output)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class[None,])
            outputs_coords.append(outputs_coord[None,])

        output = torch.cat(outputs_classes, dim=0), torch.cat(outputs_coords, dim=0)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # self attention
        self.self_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, padding_mask=None, posemb_row=None, posemb_col=None,posemb_2d=None):
        # self attention
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)

        if self.attention_type=="RCDA":
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src2 = self.self_attn((src + posemb_row).reshape(bz, h * w, c), (src + posemb_col).reshape(bz, h * w, c),
                                  src + posemb_row, src + posemb_col,
                                  src, key_padding_mask=padding_mask)[0].transpose(0, 1).reshape(bz, h, w, c)
        else:
            src2 = self.self_attn((src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  (src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  src.reshape(bz, h * w, c).transpose(0, 1), key_padding_mask=padding_mask.reshape(bz, h*w))[0].transpose(0, 1).reshape(bz, h, w, c)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0., activation="relu", n_heads=8,
                 attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # cross attention
        self.cross_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, srcs, src_padding_masks=None, adapt_pos2d=None,
                adapt_pos1d=None, posemb_row=None, posemb_col=None, posemb_2d=None):
        tgt_len = tgt.shape[1]

        query_pos = adapt_pos2d(pos2posemb2d(reference_points))
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        bz, c, h, w = srcs.shape
        srcs = srcs.permute(0, 2, 3, 1)

        if self.attention_type == "RCDA":
            query_pos_x = adapt_pos1d(pos2posemb1d(reference_points[..., 0]))
            query_pos_y = adapt_pos1d(pos2posemb1d(reference_points[..., 1]))
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src_row = src_col = srcs
            k_row = src_row + posemb_row
            k_col = src_col + posemb_col
            tgt2 = self.cross_attn((tgt + query_pos_x), (tgt + query_pos_y), k_row, k_col,
                                   srcs, key_padding_mask=src_padding_masks)[0].transpose(0, 1)
        else:
            tgt2 = self.cross_attn((tgt + query_pos).transpose(0, 1),
                                   (srcs + posemb_2d).reshape(bz, h * w, c).transpose(0,1),
                                   srcs.reshape(bz, h * w, c).transpose(0, 1), key_padding_mask=src_padding_masks.reshape(bz, h*w))[0].transpose(0,1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt


class FFN(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
    dim_t = temperature ** (2 * dim_t_)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
    dim_t = temperature ** (2 * dim_t_)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed


# build transformer
def build_transformer(cfg):
    return Transformer(
        d_model=cfg['hidden_dim'],
        nhead=cfg['num_heads'],
        num_classes=91,
        num_encoder_layers=cfg['num_encoders'],
        num_decoder_layers=cfg['num_decoders'],
        dim_feedforward=cfg['mlp_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        num_query_position=cfg['num_query_position'],
        num_query_pattern=cfg['num_query_pattern'],
        spatial_prior=cfg['spatial_prior'],
        attention_type=cfg['attention_type'],
        )
