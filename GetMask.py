import numpy as np
import torch


def get_attn_pad_mask(seq_q, seq_k):
    """
    Padding, because of unequal in source_len and target_len.
    :param seq_q: [batch, seq_len]
    :param seq_k: [batch, seq_len]
    :return:
    mask: [batch, len_q, len_k]
    """
    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch, 1, len_k], False is masked

    return pad_attn_mask.expand(batch, len_q, len_k)  # [batch, len_q, len_k]

def get_attn_subsequent_mask(seq):
    """
    Build attention mask matrix for decoder when it autoregressing.
    :param seq: [batch, target_len]
    :return: subsequent_mask: [batch, target_len, target_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch, target_len, target_len]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1) # [batch, target_len, target_len]
    subsequent_mask = torch.from_numpy(subsequent_mask)

    return subsequent_mask.byte() # [batch, target_len, target_len]

def get_pad_mask(tokens, pad_idx=0):
    """
    Get padding mask
    :param tokens: [batch, seq_len]
    :param pad_idx: int
    :return: [batch, seq_len]
    """
    batch, seq_len = tokens.size()
    pad_mask = tokens.data.eq(pad_idx).unsqueeze(1)  # [batch, 1, seq_len]
    pad_mask = pad_mask.expand(batch, seq_len, seq_len)  # [batch, seq_len, seq_len]
    return pad_mask
