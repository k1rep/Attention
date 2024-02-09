import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch, n_heads, len_q, d_k]
        :param K: [batch, n_heads, len_k, d_k]
        :param V: [batch, n_heads, len_v, d_v]
        :param attn_mask: [batch, n_heads, seq_len, seq_len]
        :return: prob: [batch, n_heads, len_q, d_v]
                 attn: [batch, n_heads, len_q, len_k]
        """
        # 1. Matmul
        scores = torch.matmul(Q, K.transpose(-1, -2)) # [batch, n_heads, len_q, len_k]

        # 2. Scale
        scores = scores / np.sqrt(self.d_k)

        if attn_mask is not None:
            # 3. Mask
            scores = scores.masked_fill(attn_mask, -np.inf)

        # 4. Softmax
        attn = nn.Softmax(dim=-1)(scores) # [batch, n_heads, len_q, len_k]

        # 5. Output
        prob = torch.matmul(attn, V) # [batch, n_heads, len_q, d_v]

        return prob, attn