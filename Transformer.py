import torch
from torch import nn
from torch import optim
from torch.utils import data

from Data import Seq2SeqDataset
from Decoder import Decoder
from Encoder import Encoder

d_model = 512 # Embedding Size
max_len = 100 # Max length of sequence
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # Dimension of K(=Q), V
n_layers = 6  # Number of Encoder and Decoder Layer
n_heads = 8  # Number of heads in Multi-Head Attention
p_drop = 0.1  # Probability of dropout

class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size=source_vocab_size, d_model=d_model, n_layers=n_layers)
        self.decoder = Decoder(target_vocab_size=target_vocab_size, d_model=d_model, n_layers=n_layers)
        self.projection = nn.Linear(d_model, target_vocab_size, bias=False)

    def forward(self, encoder_input, decoder_input):
        """
        :param encoder_input: [batch, source_len]
        :param decoder_input: [batch, target_len]
        """
        # encoder_output: [batch, source_len, d_model]
        # encoder_attns: [n_layers, batch, n_heads, source_len, source_len]
        encoder_output, encoder_attns = self.encoder(encoder_input)
        # decoder_output: [batch, target_len, d_model]
        # decoder_self_attns: [n_layers, batch, n_heads, target_len, target_len]
        # decoder_encoder_attns: [n_layers, batch, n_heads, target_len, source_len]
        decoder_output, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_input, encoder_input, encoder_output)

        # decoder_logits: [batch * target_len, target_vocab_size]
        decoder_logits = self.projection(decoder_output)
        return decoder_logits.view(-1, decoder_logits.size(-1)), encoder_attns, decoder_self_attns, decoder_encoder_attns

sentences = [
        # enc_input                dec_input            dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
source_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
source_vocab_size = len(source_vocab)

target_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(target_vocab)}
target_vocab_size = len(target_vocab)
source_len = 5 # max length of input sequence
target_len = 6

def make_data(sentences):
  encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
  for i in range(len(sentences)):
    encoder_input = [source_vocab[word] for word in sentences[i][0].split()]
    decoder_input = [target_vocab[word] for word in sentences[i][1].split()]
    decoder_output = [target_vocab[word] for word in sentences[i][2].split()]
    encoder_inputs.append(encoder_input)
    decoder_inputs.append(decoder_input)
    decoder_outputs.append(decoder_output)

  return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs)


batch_size = 64
epochs = 64
lr = 1e-3


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    encoder_inputs, decoder_inputs, decoder_outputs = make_data(sentences)
    dataset = Seq2SeqDataset(encoder_inputs, decoder_inputs, decoder_outputs)
    data_loader = data.DataLoader(dataset, 2, True)

    for epoch in range(epochs):
        """
        encoder_input: [batch, source_len]
        decoder_input: [batch, target_len]
        decoder_output: [batch, target_len]
        """
        for encoder_input, decoder_input, decoder_output in data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_output = decoder_output.to(device)

            output, encoder_attns, decoder_attns, decoder_encoder_attns = model(encoder_input, decoder_input)
            loss = criterion(output, decoder_output.view(-1))

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()