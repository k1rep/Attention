import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """ FeedForward Network """

    def __init__(self, d_model=512, d_ff=2048, p_drop=0.1, model='transformer'):
        super(FeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        if model == 'bert':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        :param x: [batch, seq_len, d_model]
        :return: [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x