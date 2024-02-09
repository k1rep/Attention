from torch.utils import data
class Seq2SeqDataset(data.Dataset):

  def __init__(self, encoder_input, decoder_input, decoder_output):
    super(Seq2SeqDataset, self).__init__()
    self.encoder_input = encoder_input
    self.decoder_input = decoder_input
    self.decoder_output = decoder_output

  def __len__(self):
    return self.encoder_input.shape[0]

  def __getitem__(self, idx):
    return self.encoder_input[idx], self.decoder_input[idx], self.decoder_output[idx]
  

class BERTDataset(data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, is_next):
        super(BERTDataset, self).__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.is_next = is_next

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.segment_ids[index], self.masked_tokens[index], self.masked_pos[index], self.is_next[index]