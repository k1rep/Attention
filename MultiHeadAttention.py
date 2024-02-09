from torch import nn

from ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_heads=8, d_model=512, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch, len_q, d_model]
        :param input_K: [batch, len_k, d_model]
        :param input_V: [batch, len_v, d_model]
        :param attn_mask: [batch, seq_len, seq_len]
        """
        residual, batch = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]


        # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
        prob, attn = ScaledDotProductAttention(d_k=self.d_k)(Q, K, V, attn_mask) # 2.当成单头注意力求输出

        # 3.Concat
        prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
        prob = prob.view(batch, -1, self.n_heads * self.d_v).contiguous() # [batch, len_q, n_heads * d_v]

        # 4.仿射变换得到最终输出
        output = self.fc(prob)  # [batch, len_q, d_model]

        return self.layer_norm(output + residual), attn