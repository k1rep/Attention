from torch import nn

from FeedForwardNetwork import FeedForwardNetwork
from MultiHeadAttention import MultiHeadAttention


class DecoderLayer(nn.Module):

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.decoder_self_attn = MultiHeadAttention()
        self.encoder_decoder_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, decoder_input, encoder_output, decoder_self_mask, decoder_encoder_mask):
        """
        :param decoder_input: [batch, target_len, d_model]
        :param encoder_output: [batch, source_len, d_model]
        :param decoder_self_mask: [batch, target_len, target_len]
        :param decoder_encoder_mask: [batch, target_len, source_len]
        :return:
        decoder_output: [batch, target_len, d_model]
        decoder_self_attn: [batch, n_heads, target_len, target_len]
        decoder_encoder_attn: [batch, n_heads, target_len, source_len]
        """
        decoder_output, decoder_self_attn = self.decoder_self_attn(decoder_input, decoder_input, decoder_input, decoder_self_mask)

        decoder_output, decoder_encoder_attn = self.encoder_decoder_attn(decoder_output, encoder_output, encoder_output, decoder_encoder_mask)
        decoder_output = self.ffn(decoder_output) # [batch, target_len, d_model]

        return decoder_output, decoder_self_attn, decoder_encoder_attn