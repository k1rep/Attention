from torch import nn

from FeedForwardNetwork import FeedForwardNetwork
from MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()
    def forward(self, encoder_input, encoder_pad_mask):
        """
        :param encoder_input: [batch, source_len, d_model]
        :param encoder_pad_mask: [batch, n_heads, source_len, source_len]
        :return:
        encoder_output: [batch, source_len, d_model]
        attn: [batch, n_heads, source_len, source_len]
        """
        encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
        encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]

        return encoder_output, attn