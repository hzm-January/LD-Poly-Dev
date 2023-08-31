# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)  # layer, 6, norm

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,  # d_model 32 nhead 2 num_encoder_layers 2
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model  # 256
        self.nhead = nhead  # 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape  # (bs,32,12,20)
        src = src.flatten(2).permute(2, 0, 1)  # (1,32,12,20) -> (1,32,240) -> (240,1,32)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (1,32,12,20) -> (1,32,240) -> (240,1,32)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (7,32) -> (7,1,32) -> (7,bs,32)

        mask = mask.flatten(1)  # (1,12,20) -> (1,240)

        tgt = torch.zeros_like(query_embed)  # (7,bs,32)

        #TODO: src不做处理，将src作为query，
        # Encoder 生成key,value

        memory, weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # mask (1,240)  pos_embed (240,1,32)
        # memory (240,bs,32) weights (bs,240,240)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)  # tgt (7,bs,32) mask (1,240) pos_embed (240,1,32) query_embed (7,bs,32) -> hs(2,7,bs,32)

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), weights  # memory(240,1,32) weights(1,240,240)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src  # (240,bs,32)

        for layer in self.layers:

            output, weights = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)  # src_key_padding_mask(1,240) pos (240,1,32)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights  # output (240,bs,32) weights (bs,240,240)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt  # tgt (7,bs,32)
        # memory (240,16,32)
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate: # return_intermediate True
            return torch.stack(intermediate)  #[(7,16,32),(7,16,32)]->(2,7,16,32)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # src(240,1,32)  pos(240,1,32)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2, weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)

        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, weights  # src(240,1,32)  pos(1,240,240)

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # d_model 32 nhead 2
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.empty_token_embedding = nn.Embedding(7, 32)
        self.max_n_lane=7

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,  # None #TODO：原代码中没有对tgt使用tgt_mask，不知为何
                     memory_mask: Optional[Tensor] = None,  # None
                     tgt_key_padding_mask: Optional[Tensor] = None,  # None #TODO：原代码中没有对tgt使用tgt_key_padding_mask，不知为何
                     memory_key_padding_mask: Optional[Tensor] = None,  # (bs,240)
                     pos: Optional[Tensor] = None,  # (240,bs,32)  #  pos 是memory的pos
                     query_pos: Optional[Tensor] = None):  # query_embed (7,bs,32)
        # tgt (7,bs,32) memory(240,bs,32) memory_key_padding_mask=mask (bs,240) pos=pos_embed (240,bs,32) query_pos=query_embed (7,bs,32) -> hs(2,7,bs,32)
        q = k = self.with_pos_embed(tgt, query_pos)  # tgt(7,bs,32) + query_pos(7,bs,32) -> (7,bs,32)
        # TODO: 针对q,q_pos自定义embedding tgt(1,bs,32) q_pos(1,bs,32)
        # tgt = torch.zeros_like(query_embed)  # (7,bs,32)
        # q_pos self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 7 32
        # TODO: 每一个query和整个memory交互信息，得到一个新的query，累加，直到生成N+1条车道线，则索引[1:]的车道线就是当前图像的检测结果
        # TODO: [1,1,32] -> [7,1,32] -> [7,bs,32]
        n_batch = tgt.shape[1]
        for batch_id in range(1, n_batch+1):  # mask (1,240)  pos_embed (240,1,32)
            # TODO: 取出当前batch id的图像特征 (240,1,32)
            latent_z = memory[:, batch_id, :][:, None]  # (240,1,32)
            memory_pos = pos[:, batch_id, :][:, None]  # (240,1,32)
            key_padding_mask_z = memory_key_padding_mask[batch_id, :][:, None]  # (bs,240)
            query_pos_cur_b = query_pos[:, batch_id, :][:, None]  # query_embed (7,bs,32)
            # TODO: 生成一个query (1,1,32)，并对query进行编码
            q_0 = self.empty_token_embedding(torch.LongTensor([batch_id]).cuda())[:, None]
            q_0_embedding = nn.Embedding(1, 32)
            q_0 = self.with_pos_embed(q_0, q_0_embedding.weight)
            lane_feats = [q_0]
            for idx in range(self.max_n_lane):
                #TODO: 取出之前生成的当前车道线的embedding，其维度为 (7,bs,32)，-> (1,1,32)
                query_pos_cur_b_l = query_pos_cur_b[idx, :, :][None]  # (1,1,32)
                q = torch.cat(lane_feats, dim=1)  # (1,k,512)

                # TODO: X进行自注意力机制  # tgt2 (7,bs,32)
                q_ = self.self_attn(q, q, value=q)[0]

                q_ = q + self.dropout1(q_)  # tgt(7,bs,32) + (7,bs,32) -> (7,bs,32)

                q_ = self.norm1(q_)  # (7,bs,32) -> (7,bs,32)

                # X = self.transformer_encoder(X, length_mask=None)  # (1,k,512) -> (1,k,512)
                # last_feat = self.transformer_decoder(latent_z, X)  # latent_z (1,1,512), X (1,k,512) -> (1,1,512)
                last_feat = self.multihead_attn(query=self.with_pos_embed(latent_z, memory_pos),  # tgt(7,bs,32)  query_pos(7,bs,32)
                                    key=self.with_pos_embed(q_, query_pos_cur_b_l),  # memory(240,bs,32)  pos(240,bs,32)
                                    value=q_, attn_mask=memory_mask,  # memory_mask None
                                    key_padding_mask=key_padding_mask_z)[0]  # memory_key_padding_mask (bs,240)
                # last_feat = self.encoders[idx](last_feat)  # (1,1,512) -> (1,1,512)
                # tuple2 - 0 attn_output (7,1,32) 1 attn_output_weight (1,7,240)
                new_q = q_ + self.dropout2(last_feat)  # (7,bs,32) -> (7,bs,32)  tgt2 (7,1,32)

                new_q = self.norm2(new_q)  # (7,bs,32) -> (7,bs,32)

                new_q_ = self.linear2(self.dropout(self.activation(self.linear1(new_q))))  # (7,bs,32) -> (7,bs,32)

                new_q_ = new_q + self.dropout3(new_q_)  # (7,bs,32) -> (7,bs,32)

                new_q_ = self.norm3(new_q_)  # (7,bs,32) -> (7,bs,32)
                # TODO: 对last_feat进行embedding
                q_embedding = nn.Embedding(1, 32)
                new_q_ = self.with_pos_embed(new_q_, q_embedding.weight)
                lane_feats.append(new_q_)

        lane_feats = torch.cat(lane_feats[1:], dim=1)  # [(1,1,512),(1,1,512),...] -> (1,K,512)

        return lane_feats # (7,bs,32)

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:  # normalize_before False
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(hidden_dim,
                      dropout,
                      nheads,
                      dim_feedforward,
                      enc_layers,
                      dec_layers,
                      pre_norm=False,
                      return_intermediate_dec=False):

    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=return_intermediate_dec,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
