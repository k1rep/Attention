from torch import nn

class Pooler(nn.Module):
    def __init__(self, d_model=768):
        super(Pooler, self).__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        :param x: [batch, seq_len, d_model]
        :return: [batch, d_model]
        """
        x = self.fc(x)
        x = self.activation(x)
        return x