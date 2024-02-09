from torch import nn

from EncoderLayer import EncoderLayer
from PositionalEncoding import PositionalEncoding
from GetMask import get_attn_pad_mask


class Encoder(nn.Module):

    def __init__(self, source_vocab_size, d_model, n_layers):
        super(Encoder, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, encoder_input):
        # encoder_input: [batch, source_len]
        encoder_output = self.source_embedding(encoder_input) # [batch, source_len, d_model]
        encoder_output = self.positional_encoding(encoder_output.transpose(0, 1)).transpose(0, 1) # [batch, source_len, d_model]

        encoder_self_attn_mask = get_attn_pad_mask(encoder_input, encoder_input) # [batch, source_len, source_len]
        encoder_self_attns = list()
        for layer in self.layers:
            # encoder_output: [batch, source_len, d_model]
            # encoder_self_attn: [batch, n_heads, source_len, source_len]
            encoder_output, encoder_self_attn = layer(encoder_output, encoder_self_attn_mask)
            encoder_self_attns.append(encoder_self_attn)

        return encoder_output, encoder_self_attns