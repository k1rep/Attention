from torch import nn
import torch

class BERTEmbedding(nn.Module):
    """ BERT Embedding layer """

    def __init__(self, max_len=30, max_vocab=50, d_model=768, n_segs=2, p_drop=0.1, device='cpu'):
        super(BERTEmbedding, self).__init__()

        self.token = nn.Embedding(max_vocab, d_model)
        self.position = nn.Embedding(max_len, d_model)
        self.segment = nn.Embedding(n_segs, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)
        self.device = device

    def forward(self, x, seg):
        """
        :param x: [batch, seq_len]
        :param seg: [batch, seq_len]
        :return: [batch, seq_len, d_model]
        """
        token_enc = self.token(x)
        pos = torch.arange(x.shape[1], dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        pos_enc = self.position(pos)
        seg_enc = self.segment(seg)
        x = self.norm(token_enc + pos_enc + seg_enc) 
        x = self.dropout(x)
        return x