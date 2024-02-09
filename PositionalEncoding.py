import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, p_drop=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

        positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # [max_len/2]
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

        # [max_len, d_model] -> [1, max_len, d_model] [max_len, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', positional_encoding)


    def forward(self, x):
        """
        :param x: [batch, seq_len, d_model]
        :return: [batch, seq_len, d_model]
        """
        x = x + self.pe[:x.size(0), ...]
        return self.dropout(x)