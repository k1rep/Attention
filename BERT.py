import random
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adadelta
from torch.utils.data import DataLoader, Dataset

from BERTEmbedding import BERTEmbedding
from BERTEncoderLayer import BERTEncoderLayer
from Data import BERTDataset
from GetMask import get_pad_mask
from Pooler import Pooler


max_len = 30
max_vocab = 50
max_pred = 5

d_k = d_v = 64
d_model = 768
d_ff = d_model * 4
n_layers = 6
n_heads = 12
n_segs = 2

p_drop = 0.1
p_mask = 0.8
p_replace = 0.1
p_do_nothing = 1 - p_mask - p_replace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)

class BERT(nn.Module):
    def __init__(self, n_layers):
        super(BERT, self).__init__()
        self.embedding = BERTEmbedding(device=device, max_len=max_len, max_vocab=max_vocab, d_model=d_model, n_segs=n_segs, p_drop=p_drop)
        self.encoders = nn.ModuleList([BERTEncoderLayer() for _ in range(n_layers)])
        self.pooler = Pooler()
        self.next_cls = nn.Linear(d_model, 2)
        self.gelu = nn.GELU()
        shared_weight = self.pooler.fc.weight
        self.fc = nn.Linear(d_model, d_model)
        self.fc.weight = shared_weight
        shared_weight = self.embedding.token.weight
        self.word_classifier = nn.Linear(d_model, max_vocab, bias=False)
        self.word_classifier.weight = shared_weight

    def forward(self, tokens, segments, masked_pos):
        output = self.embedding(tokens, segments)
        enc_self_pad_mask = get_pad_mask(tokens)
        for layer in self.encoders:
            output = layer(output, enc_self_pad_mask) # output: [batch, max_len, d_model]
        
        # NSP task
        pooled_output = self.pooler(output[:, 0]) # [batch, d_model]
        next_cls = self.next_cls(pooled_output)

        # MLM task
        # masked_pos = [batch, max_pred] -> [batch, max_pred, d_model]
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, d_model)

        # h_masked: [batch, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.gelu(self.fc(h_masked))
        logits_lm = self.word_classifier(h_masked)

        # next_cls: [batch, 2]
        # logits_lm: [batch, max_pred, max_vocab]
        return next_cls, logits_lm
    
test_text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)

# we need [MASK] [SEP] [PAD] [CLS]
word2idx = {f'[{name}]': idx for idx,
            name in enumerate(['PAD', 'CLS', 'SEP', 'MASK'])}
# {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

sentences = re.sub("[.,!?\\-]", '', test_text.lower()).split('\n')
word_list = list(set(" ".join(sentences).split()))

holdplace = len(word2idx)
for idx, word in enumerate(word_list):
    word2idx[word] = idx + holdplace

idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)
assert len(word2idx) == len(idx2word)

token_list = []
for sentence in sentences:
    token_list.append([
        word2idx[s] for s in sentence.split()
    ])

def padding(ids, n_pads, pad_symb=0):
    return ids.extend([pad_symb for _ in range(n_pads)])

def masking_procedure(cand_pos, input_ids, masked_symb=word2idx['[MASK]']):
    masked_pos = []
    masked_tokens = []
    for pos in cand_pos:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        if random.random() < p_mask:
            input_ids[pos] = masked_symb
        elif random.random() > (p_mask + p_replace):
            rand_word_idx = random.randint(4, vocab_size - 1)
            input_ids[pos] = rand_word_idx

    return masked_pos, masked_tokens

def make_data(sentences, n_data):
    batch_data = []
    positive = negative = 0
    len_sentences = len(sentences)
    # 50% sampling adjacent sentences, 50% sampling not adjacent sentences
    while positive != n_data / 2 or negative != n_data / 2:
        tokens_a_idx = random.randrange(len_sentences)
        tokens_b_idx = random.randrange(len_sentences)
        tokens_a = sentences[tokens_a_idx]
        tokens_b = sentences[tokens_b_idx]

        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0 for i in range(
            1 + len(tokens_a) + 1)] + [1 for i in range(1 + len(tokens_b))]

        n_pred = min(max_pred, max(1, int(len(input_ids) * .15)))
        cand_pos = [i for i, token in enumerate(input_ids)
                    if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]

        # shuffle all candidate position index, to sampling maksed position from first n_pred
        masked_pos, masked_tokens = masking_procedure(
            cand_pos[:n_pred], input_ids, word2idx['[MASK]'])

        # zero padding for tokens
        padding(input_ids, max_len - len(input_ids))
        padding(segment_ids, max_len - len(segment_ids))

        # zero padding for mask
        if max_pred > n_pred:
            n_pads = max_pred - n_pred
            padding(masked_pos, n_pads)
            padding(masked_tokens, n_pads)

        if (tokens_a_idx + 1) == tokens_b_idx and positive < (n_data / 2):
            batch_data.append(
                [input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif (tokens_a_idx + 1) != tokens_b_idx and negative < (n_data / 2):
            batch_data.append(
                [input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch_data

batch_size = 6
batch_data = make_data(token_list, n_data=batch_size)
batch_tensor = [torch.LongTensor(ele) for ele in zip(*batch_data)]

dataset = BERTDataset(*batch_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = BERT(n_layers)
lr = 1e-3
epochs = 500
criterion = nn.CrossEntropyLoss()
optimizer = Adadelta(model.parameters(), lr=lr)
model.to(device)

# training
for epoch in range(epochs):
    for one_batch in dataloader:
        input_ids, segment_ids, masked_tokens, masked_pos, is_next = [ele.to(device) for ele in one_batch]

        logits_cls, logits_lm = model(input_ids, segment_ids, masked_pos)
        loss_cls = criterion(logits_cls, is_next)
        loss_lm = criterion(logits_lm.view(-1, max_vocab), masked_tokens.view(-1))
        loss_lm = (loss_lm.float()).mean()
        loss = loss_cls + loss_lm
        if (epoch + 1) % 10 == 0:
            print(f'Epoch:{epoch + 1} \t loss: {loss:.6f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Using one sentence to test
test_data_idx = 3
model.eval()
with torch.no_grad():
    input_ids, segment_ids, masked_tokens, masked_pos, is_next = batch_data[test_data_idx]
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(device)
    masked_pos = torch.LongTensor(masked_pos).unsqueeze(0).to(device)
    masked_tokens = torch.LongTensor(masked_tokens).unsqueeze(0).to(device)
    logits_cls, logits_lm = model(input_ids, segment_ids, masked_pos)
    input_ids, segment_ids, masked_tokens, masked_pos, is_next = batch_data[test_data_idx]
    print("========================================================")
    print("Masked data:")
    masked_sentence = [idx2word[w] for w in input_ids if idx2word[w] != '[PAD]']
    print(masked_sentence)

    # logits_lm: [batch, max_pred, max_vocab]
    # logits_cls: [batch, 2]
    cpu = torch.device('cpu')
    pred_mask = logits_lm.data.max(2)[1][0].to(cpu).numpy()
    pred_next = logits_cls.data.max(1)[1].data.to(cpu).numpy()[0]

    bert_sentence = masked_sentence.copy()
    original_sentence = masked_sentence.copy()

    for i in range(len(masked_pos)):
        pos = masked_pos[i]
        if pos == 0:
            break
        bert_sentence[pos] = idx2word[pred_mask[i]]
        original_sentence[pos] = idx2word[masked_tokens[i]]

    print("BERT reconstructed:")
    print(bert_sentence)
    print("Original sentence:")
    print(original_sentence)

    print("===============Next Sentence Prediction===============")
    print(f'Two sentences are continuous? {True if is_next else False}')
    print(f'BERT predict: {True if pred_next else False}')