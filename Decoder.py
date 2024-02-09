import torch
from torch import nn

from DecoderLayer import DecoderLayer
from PositionalEncoding import PositionalEncoding
from GetMask import get_attn_pad_mask, get_attn_subsequent_mask


class Decoder(nn.Module):

    def __init__(self, target_vocab_size, d_model, n_layers):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, decoder_input, encoder_input, encoder_output):
        """
        :param decoder_input: [batch, target_len]
        :param encoder_input: [batch, source_len]
        :param encoder_output: [batch, source_len, d_model]
        """
        decoder_output = self.target_embedding(decoder_input) # [batch, target_len, d_model]
        decoder_output = self.positional_encoding(decoder_output.transpose(0, 1)).transpose(0, 1) # [batch, target_len, d_model]
        decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input) # [batch, target_len, target_len]
        decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input) # [batch, target_len, target_len]

        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input) # [batch, target_len, source_len]

        decoder_self_mask = torch.gt((decoder_self_attn_mask + decoder_subsequent_mask), 0) # [batch, target_len, target_len]
        decoder_self_attns, decoder_encoder_attns = [], []

        for layer in self.layers:
            # decoder_output: [batch, target_len, d_model]
            # decoder_self_attn: [batch, n_heads, target_len, target_len]
            # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
            decoder_output, decoder_self_attn, decoder_encoder_attn = layer(decoder_output, encoder_output, decoder_self_mask, decoder_encoder_attn_mask)
            decoder_self_attns.append(decoder_self_attn)
            decoder_encoder_attns.append(decoder_encoder_attn)

        return decoder_output, decoder_self_attns, decoder_encoder_attns