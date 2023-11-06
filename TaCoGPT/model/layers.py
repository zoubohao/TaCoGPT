import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    lineage_n: int = 6
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6
    dropout_prob: float
    if_attention_nolinear: bool
    pad_id: int = 0
    pretrain_TaCoGPT: bool = False
    cache_infer: bool
    cache_infer_gpt: bool
    k_mer_k: int
    # auto
    k_vocab_size: int
    a_vocab_size: int
    model_max_seq_len: int
    seq_max_len: int


def empty_cache(model: nn.Module):
    if hasattr(model, "cache_seq_norm_rep"):
        model.cache_seq_norm_rep = None  # for TaCoGPT
    if hasattr(model, "cache_input_tensor"):
        model.cache_input_tensor = None  # for GPT
        model.cache_idx = None
    # for attention modules
    for m in model.modules():
        if hasattr(m, "cache_v") or hasattr(m, "cache_k"):
            m.cache_v = None
            m.cache_k = None
        if hasattr(m, "cache_seq_norm_rep"):
            m.cache_seq_norm_rep = None


def alibi(score: torch.Tensor):
    """
    generateing alibi attention mask for training short and testing long.
    Args:
        score (torch.Tensor): the score tensor, [b, h, l, l]

    Returns:
        mask: (torch.Tensor): [l, l]
    """
    len_seq = score.size(-1)
    m = torch.arange(0, len_seq).expand(len_seq, len_seq)
    n = m.clone().T
    mask = -torch.abs(m - n)
    return mask.tril(diagonal=0)


def get_attention_padding_mask(inputs, pad_id=0):
    attn_pad_mask = inputs.eq(pad_id).unsqueeze(1).repeat(1, inputs.size(1), 1)
    # |attn_pad_mask| : (batch_size, q_len, k_len)
    return attn_pad_mask


def get_attention_subsequent_mask(inputs):
    bs, q_len = inputs.size()
    subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
    # |subsequent_mask| : (batch_size, q_len, q_len)
    return subsequent_mask


def get_autoregression_mask(inputs, pad_id=0):
    attn_pad_mask = get_attention_padding_mask(inputs, pad_id)
    subsequent_mask = get_attention_subsequent_mask(inputs).to(device=attn_pad_mask.device)
    attn_mask = torch.gt((attn_pad_mask.to(dtype=subsequent_mask.dtype) +
                         subsequent_mask), 0).float() * -100000.
    return attn_mask.unsqueeze(1)


def padding_only_mask(tokens: torch.Tensor, pad_id=0):
    """
    Return the padding mask just for pretraining model by using DNA sequences.

    Args:
        tokens (torch.Tensor): the tensor with shape [batch_size, seq_len]
        pad_id (int): the padding index. Default 0

    Returns:
        torch.Tensor: the padding mask with shape [batch, 1, seq_len, seq_len]
    """
    attn_pad_mask = tokens.eq(pad_id).unsqueeze(1).repeat(1, tokens.size(1), 1).float() * -100000.
    # |attn_pad_mask| : (batch_size, q_len, k_len)
    return attn_pad_mask.unsqueeze(1)


def padding_mix_generate_mask(tokens: torch.Tensor, start_pos_no_see: int, pad_id=0):
    """
    Return the mask for fine-tuning the model with annotation labels.

    Args:
        tokens (torch.Tensor): the tensor with shape [batch_size, seq_len]
        start_pos_no_see (int): The start position of annotation labels. Subsequent after the DNA sequences. It would use <s> as start.
        pad_id (int, optional): the padding index. Defaults to 0.

    Returns:
        torch.Tensor: the padding mask with shape [batch, 1, seq_len, seq_len]
    """

    def get_attention_padding_mask(q, k, pad_id: int):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1).float() * -100000.
        # |attn_pad_mask| : (batch_size, q_len, k_len)
        return attn_pad_mask

    def get_attention_mixed_mask(q, start_pos_no_see):
        bs, q_len = q.size()
        assert q_len >= start_pos_no_see, "The length of tokens smaller than start position of generating."
        upper_mask = torch.zeros(bs, start_pos_no_see, q_len)
        upper_mask[:, :, start_pos_no_see:] = -100000

        bottom_mask = torch.ones(bs, q_len - start_pos_no_see,
                                 q_len).triu(diagonal=start_pos_no_see + 1) * -100000.0
        return torch.cat([upper_mask, bottom_mask], dim=1).to(q.device)

    bs, q_len = tokens.size()
    # (batch_size, seq_len)
    attn_pad_mask = get_attention_padding_mask(tokens, tokens, pad_id)
    # |attn_pad_mask| : (batch_size, seq_len, seq_len)

    subsequent_mask = get_attention_mixed_mask(tokens, start_pos_no_see)
    # |subsequent_mask| : (batch_size, seq_len, seq_len)

    attn_mask = torch.zeros(size=[bs, q_len, q_len], device=tokens.device).masked_fill(
        (attn_pad_mask + subsequent_mask) < 0, -100000.0
    )
    # |attn_mask| : (batch_size, seq_len, seq_len)
    return attn_mask.unsqueeze(1)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# RoPE rotation relative positional embedding
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.pretrain_TaCoGPT = args.pretrain_TaCoGPT
        self.n_local_heads = args.n_heads // 1
        self.head_dim = args.dim // args.n_heads
        self.cache_infer = args.cache_infer
        self.cache_infer_gpt = args.cache_infer_gpt

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        self.dropout_atten = nn.Dropout(p=args.dropout_prob)
        self.dropout_out = nn.Dropout(p=args.dropout_prob)

        self.cache_k = None
        self.cache_v = None

    @torch.inference_mode()
    def infer_cache(self, x, freqs_cis, mask):
        """_summary_

        Args:
            x (torch.Tensor): first x [b, model_max_len + 1, c], after [b, 1, c]
            freqs_cis (torch.Tensor): _description_
            mask (Optional[torch.Tensor]): _description_

        Returns:
            _type_: _description_
        """
        bsz, seqlen, _ = x.shape
        mask_len = mask.size(-1)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.cache_k is None:
            # first case
            self.cache_k = xk.clone().detach()
            self.cache_v = xv.clone().detach()
            keys = self.cache_k
            values = self.cache_v
        else:
            # after case
            if self.cache_infer_gpt:
                # for generating DNA sequence by using GPT (beam searching)
                cur_cache_k = self.cache_k
                cur_cache_v = self.cache_v
                keys = torch.cat([cur_cache_k, xk], dim=1)
                values = torch.cat([cur_cache_v, xv], dim=1)
            else:
                # for generating taxonomic class by using TaCoGPT
                cur_cache_k = self.cache_k[:, 0: (mask_len - 1)]
                cur_cache_v = self.cache_v[:, 0: (mask_len - 1)]
                keys = torch.cat([cur_cache_k, xk], dim=1)
                values = torch.cat([cur_cache_v, xv], dim=1)
                self.cache_k = keys
                self.cache_v = values

        xq = xq.transpose(1, 2)  # [b, h_num, 1, dim]
        keys = keys.transpose(1, 2)  # [b, h_num, model_len + seq_len, dim]
        values = values.transpose(1, 2)  # [b, h_num, model_len + seq_len, dim]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print("score shape: ", scores.shape, "mask shape: ", mask.shape)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def forward(self, x, freqs_cis, mask):
        if self.cache_infer and self.training is False and self.pretrain_TaCoGPT is False:
            return self.infer_cache(x, freqs_cis, mask)

        bsz, seqlen, _ = x.shape  # [batch, seq_len, dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # print(mask)

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, slen)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout_atten(scores)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.dropout_out(self.wo(output))


class GateActivationFunction(nn.Module):

    def __init__(self, in_dim) -> None:
        super().__init__()
        self.gate = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, p: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.drop_out = nn.Dropout(p)
        self.drop_m = nn.Dropout(p)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.drop_m(x)
        x = self.w2(x)
        return self.drop_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim,
                                        multiple_of=args.multiple_of, p=args.dropout_prob)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.gate = nn.Parameter(torch.ones(
            size=[2], dtype=torch.float32, requires_grad=True), requires_grad=True)

    def forward(self, x, freqs_cis, mask):
        h = x * self.gate[0] + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        out = h * self.gate[1] + self.feed_forward.forward(self.ffn_norm(h))
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False
        )
        self.dwd = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=3, groups=in_channels, bias=False, dilation=3
        )
        self.ln = nn.LayerNorm(in_channels, eps=1e-6)
        self.point = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.stridec_conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): B, C, L

        Returns:
            _type_: _description_
        """
        conv = self.dwd(self.dw(x)).transpose(1, 2)
        conv = self.ln(conv).transpose(1, 2)
        h = self.point(conv) + x
        return self.stridec_conv(h) + self.c1(h)


class DyConv(nn.Module):
    def __init__(self, dim: int, model_max_len: int) -> None:
        super().__init__()
        self.model_max_len = model_max_len
        self.conv1 = DownSample(dim, dim)
        self.conv2 = DownSample(dim, dim)
        self.conv3 = DownSample(dim, dim)

    def forward(self, x, token_k):
        """
        x: B, C, L
        token_K: B, L
        """
        b, c, l = x.shape
        device = x.device
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # getting mask
        with torch.no_grad():
            indices = []
            for i in range(b):
                cur_index = torch.nonzero(token_k[i], as_tuple=True)[0][-1].item()
                indices.append(cur_index + 1)
            mask_tokens = []
            for i, index in enumerate(indices):
                # calculate the stop point.
                true_l = index // 8
                # get useful tokens number.
                if true_l <= self.model_max_len:
                    true_l += 25
                    if true_l > self.model_max_len:
                        true_l = self.model_max_len

                # padding zeros for mask.
                a = torch.ones(size=[1, true_l], dtype=torch.long, device=device)
                if true_l < self.model_max_len:
                    with torch.no_grad():
                        padding_num = self.model_max_len - true_l
                        b = torch.zeros(size=[1, padding_num], dtype=torch.long, device=device)
                    a = torch.cat([a, b], dim=-1)
                mask_tokens.append(a)
            mask_tensor = torch.cat(mask_tokens, dim=0)

        return x, mask_tensor
