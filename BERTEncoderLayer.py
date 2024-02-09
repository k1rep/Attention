from torch import nn
from FeedForwardNetwork import FeedForwardNetwork
from MultiHeadAttention import MultiHeadAttention


class BERTEncoderLayer(nn.Module):
    """ BERT Encoder Layer """
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super(BERTEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.encoder_self_attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model, d_k=d_k, d_v=d_v)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_model*4, model='bert')
    def forward(self, encoder_input, encoder_pad_mask):
        residual = encoder_input
        encoder_input = self.norm1(encoder_input)
        output, _ = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
        encoder_input = output + residual
        residual = encoder_input
        encoder_input = self.norm2(encoder_input)
        encoder_input = self.ffn(encoder_input) + residual
        return encoder_input