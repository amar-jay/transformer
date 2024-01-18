import torch
import re
import json
import torch.nn as nn
import torch.nn.functional as F

class Config:
    n_embd = 1000 # embedding dimensionality
    block_size = 128 # length of the input sequences of integers, context length
    batch_size = 4
    dropout = 0.2
    vocab_size=83
    device='cuda' if torch.cuda.is_available() else 'cpu'


# fetch poems based on poem file path
def fetch_data(filename:str):
    turkish_pattern = re.compile(r'^[0-9a-zA-ZâçÇğĞıİöÖşŞüÜ\-\,\"\'\!\?\.\n\s]+$')
    valid_titles = [] # list of valid poem titles
    full_text = ""

    with open(filename, "r") as f:
        data = json.loads(f.read())
        for i in range(len(data)):
            title = data[i]['baslik'].strip().split()
            title = " ".join(title) # remove multiple spaces             

            author = data[i]['sair'].strip().split()
            author = " ".join(author)

            content = data[i]['icerik']
            content = " ".join(content[1:-1])

            if re.match(turkish_pattern, title) and re.match(turkish_pattern, author):
                valid_titles.append(title)
            if re.match(turkish_pattern, content):
                full_text += content + ".\n"
    return full_text, valid_titles

# creates encode, decode,functinos and returns vocab size
def text_encoder_builder(sample_text:str):
    vocabs = sorted(list(set(sample_text)))
    vocab_size = len(vocabs)
    itoa = dict(enumerate(vocabs))
    atoi = {c: i for i, c in itoa.items()}
    encode = lambda chars: [atoi[c] for c in chars]
    decode = lambda chars: [itoa[c] for c in chars]

    return encode, decode, vocab_size


# creates an infinte batch sampler
def create_batch_sampler(data, config:Config, split=(0.9,0.99)):
    n1 = int(split[0] * len(data))
    n2 = int(split[1] * len(data))
    split = {
            'train': data[:n1],
            'val': data[n1:n2],
            'test': data[n2:],
            }
    def get_batch(dataset='train'): # fetching batches randomly
        dataset = split[dataset]
        idx = torch.randint(0, len(dataset)-config.block_size-1, (config.batch_size,))
        x = [dataset[i: i+config.block_size] for i in idx]
        y = [dataset[i+1: i+config.block_size+1] for i in idx]
        return torch.stack(x), torch.stack(y)
    return get_batch


# NOTE: cross check first
class MaskedAttention(nn.Module):
    def __init__(self, config:Config):
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.out = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        x = self.key(x) @ self.query(x).transpose(-2, -1)
        tril = torch.tril(torch.ones_like(x))
        x.masked_fill(tril == 0, float('-inf'))
        x = F.softmax(x, dim=-1)

        x = x.view(B, T, C)
        x = x @ self.value(x).transpose(-2, -1)

        return self.out(x) 


# a decoder transformer model
class Decoder(nn.Module):
    def __init__(self, config:Config):
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=config.device)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd, device=config.device)
        self.masked_att = MaskedAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, 4 * config.n_embd)
            )
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, config.vocab_size, bias=False, device=config.device)
    def forward(self, x):
        tok = self.tok_emb(x)
        pos = self.pos_emb(x)
        x = tok + pos

        # add and norm or norm and add. Which comes first?
        x = x + self.masked_att(tok + pos) # res net with attention
        x = self.ln_1(x)
        x = x + self.mlp(x)
        x = self.ln_2(x)
        # normalize it

        return self.out(x)

