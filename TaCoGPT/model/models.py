import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from TaCoGPT.model.layers import (DyConv, ModelArgs, RMSNorm, TransformerBlock,
                                  get_autoregression_mask,
                                  padding_mix_generate_mask,
                                  precompute_freqs_cis)


# def nan_check_and_replace(tensor: torch.Tensor):
#     nan_check = torch.isnan(tensor).int()
#     if nan_check.sum() != 0:
#         return torch.where(nan_check == 1, 0., tensor)
#     return tensor


class TaCoGPT(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.pad_id = params.pad_id
        self.model_max_seq_len = params.model_max_seq_len
        self.dim = params.dim
        self.seq_max_len = params.seq_max_len
        self.if_pretrain = params.pretrain_TaCoGPT

        self.tok_embeddings_k = nn.Embedding(params.k_vocab_size, params.dim, padding_idx=0)

        self.seg_embeddings_k = nn.Parameter(
            torch.randn(size=[1, params.model_max_seq_len + 10, params.dim],
                        pin_memory=True, requires_grad=True),
            requires_grad=True,
        )
        self.seg_embeddings_a = nn.Parameter(
            torch.randn(size=[1, 8, params.dim], pin_memory=True, requires_grad=True), requires_grad=True
        )
        nn.init.normal_(self.seg_embeddings_k, 1, 0.25)
        nn.init.normal_(self.seg_embeddings_a, -1, 0.25)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.model_max_seq_len * 2)
        self.logit_scale = nn.Parameter(torch.ones(
            [], requires_grad=True) * np.log(1 / 0.07), requires_grad=True)

        self.conv = DyConv(params.dim, params.model_max_seq_len)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        if self.if_pretrain:
            self.tokens_a_placer = nn.Parameter(data=torch.randn([7, params.dim],
                                                                 requires_grad=True),
                                                requires_grad=True)
        else:
            self.a_vocab_size = params.a_vocab_size
            self.tok_embeddings_a = nn.Embedding(params.a_vocab_size, params.dim, padding_idx=0)
            self.lstm = nn.LSTM(input_size=params.dim, hidden_size=params.dim // 2, num_layers=5)
            self.feature = nn.Linear(params.dim // 2, params.dim, bias=False)
            self.output_a = nn.Linear(params.dim, self.a_vocab_size, bias=False)

        self.cache_seq_norm_rep = None

    def gatherValues(self, v1, t2):
        """
        t1 = torch.randn([3, 4])
        t2 = torch.randn([3, 5, 4])
        print(gatherValues(t1, t2, 5))
        t2v = t2.view(15, 4)
        print(t1 @ t2v.T)
        v1: [b, dim]
        t2: [b, num_labels, dim]
        return: [b, num_labels]
        """
        b1 = v1.size(0)
        b2 = t2.size(0)
        assert b1 == b2, ValueError("Batch size is not equal.")
        num_labels = t2.size(1)
        dotTensor = torch.tensordot(v1, t2, dims=([1], [2])).permute([
            0, 2, 1])  # [b1, num_labels, b2]
        index = torch.arange(b1).expand([num_labels, b1]).transpose(
            1, 0).unsqueeze(-1).to(dotTensor.device)
        return torch.gather(dotTensor, dim=-1, index=index).squeeze(-1)

    def lineage_encoder_norm(self, lineage_tensor):
        contrast_lineages = self.tok_embeddings_a(lineage_tensor).permute([1, 0, 2])  # L, B, C
        x, _ = self.lstm(contrast_lineages)
        x = x[-1]
        lineages_rep = self.feature(x)  # [B*num_labels, C]
        lineages_rep = lineages_rep / lineages_rep.norm(dim=-1, keepdim=True)
        return lineages_rep

    def seq_encoder_norm(self, seq_tensor, tokens_k_mask):
        seq_tokens_rep = seq_tensor
        tokens_k_mask = (1.0 - tokens_k_mask.unsqueeze(-1)) * -1000000.0  # [B, L, 1]
        sft = torch.softmax(torch.mean(seq_tokens_rep, dim=-1, keepdim=True) + tokens_k_mask, dim=1)
        # print(sft)
        seq_rep = torch.sum(seq_tokens_rep * sft, dim=1)  # [B, C]
        seq_rep = seq_rep / seq_rep.norm(dim=-1, keepdim=True)
        return seq_rep

    def _prepare_pretrain(self, tokens_k, device):
        input_seq_tensor = self.tok_embeddings_k(tokens_k)  # b, l, c
        input_seq_tensor = torch.permute(input_seq_tensor, [0, 2, 1])
        input_seq_tensor, tokens_k_mask = self.conv(input_seq_tensor, tokens_k)
        input_seq_tensor = torch.permute(input_seq_tensor, [0, 2, 1])  # b, l, c

        b, model_seq_k_len = tokens_k_mask.shape
        lineage_a_len = self.tokens_a_placer.size(0)
        tokens_a = torch.ones([b, lineage_a_len], dtype=torch.long, device=device)

        assert input_seq_tensor.size(1) == model_seq_k_len

        seqlen = model_seq_k_len + lineage_a_len
        freqs_cis = self.freqs_cis[0:seqlen]
        freqs_cis = freqs_cis.to(device)
        tokens_g = torch.cat([tokens_k_mask, tokens_a], dim=-1)
        mask_mix = padding_mix_generate_mask(tokens_g, model_seq_k_len + 1, self.pad_id)
        mask_mix = mask_mix.to(device)
        seg_k = self.seg_embeddings_k[:, 0:model_seq_k_len, :]
        input_taxon_tensor = self.tokens_a_placer.expand([b, lineage_a_len, -1])
        input_taxon_tensor = input_taxon_tensor + torch.zeros_like(
            input_taxon_tensor, dtype=torch.float32, requires_grad=True
        )
        seg_a = self.seg_embeddings_a[:, 0:lineage_a_len, :]
        h_g = torch.cat([input_seq_tensor, input_taxon_tensor], dim=1)
        seg_f = torch.cat([seg_k, seg_a], dim=1)

        return h_g, seg_f, tokens_k_mask, freqs_cis, mask_mix, model_seq_k_len

    def pretrain(self, X: torch.Tensor):
        """_summary_

        Args:
            X (torch.Tensor): the sequence indices and lineages indices, (2 * b, seq_max_len), we would add lineage placer,
            X1 = X[0: b]
            X2 = X[b: 2b]
        Returns:
            _type_: _description_
        """
        seq_max_len = self.seq_max_len
        device = X.device
        b_two = X.size(0)
        assert b_two % 2 == 0
        b = b_two // 2

        assert X.size(-1) == seq_max_len
        h_g, seg_f, tokens_k_mask, freqs_cis, mask_mix, model_seq_k_len = self._prepare_pretrain(
            X, device)
        for i in range(self.n_layers):
            h_g += seg_f
            h_g = self.layers[i](h_g, freqs_cis, mask_mix)
        h_g = self.norm(h_g)
        # contrast
        h_g_ori = h_g.clone()
        seq_tokens_rep = h_g_ori[:, 0:model_seq_k_len, :]
        seq_rep_norm = self.seq_encoder_norm(seq_tokens_rep, tokens_k_mask)
        x1_seq_rep_norm = seq_rep_norm[0:b]
        x2_seq_rep_norm = seq_rep_norm[b:]
        pred_logit = (x1_seq_rep_norm * x2_seq_rep_norm).sum(-1)
        # print(pred_logit)
        pred_logit *= self.logit_scale.exp()
        return pred_logit

    def _prepare(self, tokens_k, tokens_a, device):
        input_seq_tensor = self.tok_embeddings_k(tokens_k)  # b, l, c
        input_seq_tensor = torch.permute(input_seq_tensor, [0, 2, 1])
        input_seq_tensor, tokens_k_mask = self.conv(input_seq_tensor, tokens_k)
        input_seq_tensor = torch.permute(input_seq_tensor, [0, 2, 1])  # b, l, c
        _, model_seq_k_len = tokens_k_mask.shape
        _, lineage_a_len = tokens_a.shape

        assert input_seq_tensor.size(1) == model_seq_k_len

        seqlen = model_seq_k_len + lineage_a_len
        freqs_cis = self.freqs_cis[0:seqlen]
        freqs_cis = freqs_cis.to(device)
        tokens_g = torch.cat([tokens_k_mask, tokens_a], dim=-1)
        mask_mix = padding_mix_generate_mask(tokens_g, model_seq_k_len + 1, self.pad_id)
        mask_mix = mask_mix.to(device)

        # print("lineage length: ", tokens_a.size(-1))
        # print("tokens_g model_seq_k_len: ", tokens_g[0, (model_seq_k_len-5):])
        # print("mask: ", mask_mix[0, 0, (model_seq_k_len-5):, (model_seq_k_len-5):])

        seg_k = self.seg_embeddings_k[:, 0: model_seq_k_len, :]
        input_taxon_tensor = self.tok_embeddings_a(tokens_a)
        # input_taxon_tensor = input_taxon_tensor + torch.zeros_like(
        #     input_taxon_tensor, dtype=torch.float32, requires_grad=True
        # )
        seg_a = self.seg_embeddings_a[:, 0: lineage_a_len, :]
        h_g = torch.cat([input_seq_tensor, input_taxon_tensor], dim=1)
        seg_f = torch.cat([seg_k, seg_a], dim=1)

        return h_g, seg_f, tokens_k_mask, freqs_cis, mask_mix, model_seq_k_len

    def finetune_train(self, X: torch.Tensor, contrast_lineages: torch.Tensor):
        """_summary_

        Args:
            X (torch.Tensor): the sequence indices and lineages indices, (B, seq_max_len + lineage_len)
            contrast_lineages: [b, num_neg, 6]
        Returns:
            _type_: _description_
        """
        seq_max_len = self.seq_max_len
        device = X.device
        assert X.size(-1) > seq_max_len
        tokens_k = X[:, 0:seq_max_len]
        tokens_a = X[:, seq_max_len:]

        h_g, seg_f, tokens_k_mask, freqs_cis, mask_mix, model_seq_k_len = \
            self._prepare(tokens_k, tokens_a, device)

        for i in range(self.n_layers):
            h_g += seg_f
            h_g = self.layers[i](h_g, freqs_cis, mask_mix)

        h_g = self.norm(h_g)
        h_g_ori = h_g.clone()
        logit_scale = self.logit_scale.exp()

        phy_tokens = h_g[:, model_seq_k_len, :]  # b, c
        pred_phy = self.output_a(phy_tokens)

        h_g = self.output_a(h_g)
        predict_output = h_g[:, model_seq_k_len:-1, :]  # for tokens_a[1:]

        # lineages representations
        b, num_labels, lineageLength = contrast_lineages.shape
        contrast_lineages = contrast_lineages.contiguous().view([b * num_labels, lineageLength])
        lineages_rep_norm = self.lineage_encoder_norm(contrast_lineages)
        lineages_rep_norm = lineages_rep_norm.view(b, num_labels, -1)

        # sequence representations
        seq_tokens_rep = h_g_ori[:, 0:model_seq_k_len, :]
        seq_rep_norm = self.seq_encoder_norm(seq_tokens_rep, tokens_k_mask)

        # predict contarstive learning
        predict_contrast = self.gatherValues(seq_rep_norm, lineages_rep_norm) * logit_scale

        return predict_output, pred_phy, predict_contrast

    @torch.inference_mode()
    def finetune_infer(self, X: torch.Tensor):
        seq_max_len = self.seq_max_len
        device = X.device
        if X.size(-1) == seq_max_len + 1:
            the_first_time = True
        else:
            the_first_time = False

        tokens_k = X[:, 0:seq_max_len]
        tokens_a = X[:, seq_max_len:]

        h_g, seg_f, tokens_k_mask, freqs_cis, mask_mix, model_seq_k_len = self._prepare(
            tokens_k, tokens_a, device)

        if the_first_time:
            # cache the seq representation tensor
            for i in range(self.n_layers):
                h_g += seg_f
                h_g = self.layers[i](h_g, freqs_cis, mask_mix)

            h_g = self.norm(h_g)
            h_g_ori = h_g.clone()

            predict_next = self.output_a(h_g)[:, -1, :]  # [B, a_vocab_size]
            predict_next = torch.softmax(predict_next, dim=-1)

            # sequence representations
            seq_tokens_rep = h_g_ori[:, 0:model_seq_k_len, :]
            seq_rep_norm = self.seq_encoder_norm(seq_tokens_rep, tokens_k_mask)
            self.cache_seq_norm_rep = seq_rep_norm
            return predict_next
        else:
            cur_token_a_size = tokens_a.size(1)
            idx = model_seq_k_len + cur_token_a_size - 1
            mask_next = mask_mix[:, :, idx:, :]
            h_g_next = h_g[:, idx:, :]
            seg_f_next = seg_f[:, idx:, ]
            freqs_cis_next = freqs_cis[idx:]

            for i in range(self.n_layers):
                h_g_next += seg_f_next
                h_g_next = self.layers[i](h_g_next, freqs_cis_next, mask_next)

            h_g_next = self.norm(h_g_next)
            predict_next = self.output_a(h_g_next)[:, -1, :]  # [B, a_vocab_size]
            predict_next = torch.softmax(predict_next, dim=-1)
            return predict_next

    def forward(self, X: torch.Tensor, contrast_lineages: torch.Tensor = None):
        if self.if_pretrain:
            return self.pretrain(X)
        else:
            if self.training:
                return self.finetune_train(X, contrast_lineages)
            else:
                return self.finetune_infer(X)


class GPT(nn.Module):

    def __init__(self, params: ModelArgs) -> None:
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.pad_id = params.pad_id
        self.model_max_seq_len = params.model_max_seq_len
        self.dim = params.dim
        self.seq_max_len = params.seq_max_len
        self.n_heads = params.n_heads
        self.k_mer_num = params.k_mer_k
        self.cache_infer = params.cache_infer

        assert params.dim % self.k_mer_num == 0, ValueError("dim_of_model % k_mer_k == 0 !")

        self.tok_embeddings_k = nn.Embedding(params.k_vocab_size, params.dim, padding_idx=0)
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.model_max_seq_len * 2)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.tokens_k_out = nn.Linear(params.dim, params.k_vocab_size, bias=False)
        self.cache_input_tensor = None
        self.cache_idx = None

    def forward_train(self, tokens_k: torch.Tensor):
        """
        Args:
            tokens_k (torch.Tensor): [b, seq_len]
        """
        model_seq_k_len = tokens_k.size(-1)
        device = tokens_k.device
        input_seq_tensor = self.tok_embeddings_k(tokens_k)  # b, l * kmer, c

        freqs_cis = self.freqs_cis[0: model_seq_k_len]
        freqs_cis = freqs_cis.to(device)

        mask = get_autoregression_mask(tokens_k, self.pad_id)

        h_g = input_seq_tensor.clone()
        for i in range(self.n_layers):
            h_g = self.layers[i](h_g, freqs_cis, mask)

        h_g = self.norm(h_g)  # [b, l, c]

        return self.tokens_k_out(h_g)  # the output of last token would be omited.

    def forward_infer(self, tokens_k: torch.Tensor):
        """
        Args:
            tokens_k (torch.Tensor): [b, l]
        """
        bsz, seqlen = tokens_k.shape
        device = tokens_k.device
        if self.cache_input_tensor is None:
            input_seq_tensor = self.tok_embeddings_k(tokens_k)  # b, l, c
        else:
            input_seq_tensor = self.tok_embeddings_k(tokens_k[:, self.cache_idx:])  # b, k, c
            input_seq_tensor = torch.cat([self.cache_input_tensor, input_seq_tensor], dim=1)

        freqs_cis = self.freqs_cis[0:seqlen].to(device)
        mask = get_autoregression_mask(tokens_k, self.pad_id)
        h_g = input_seq_tensor.clone()

        if self.cache_input_tensor is not None and self.cache_idx is not None:
            h_g = h_g[:, self.cache_idx:, :]
            freqs_cis = freqs_cis[self.cache_idx:]
            mask = mask[:, :, self.cache_idx:, :]

        for i in range(self.n_layers):
            h_g = self.layers[i](h_g, freqs_cis, mask)

        h_g = self.norm(h_g)  # [b, l, c]
        h_g_out = self.tokens_k_out(h_g)

        if self.cache_input_tensor is None and self.cache_idx is None:
            self.cache_input_tensor = input_seq_tensor
            self.cache_idx = seqlen
        return h_g_out

    def forward(self, tokens_k):
        if self.training is False and self.cache_infer:
            return self.forward_infer(tokens_k)
        return self.forward_train(tokens_k)
