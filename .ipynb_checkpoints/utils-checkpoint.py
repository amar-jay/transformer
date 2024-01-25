import torch
import re
import json

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
