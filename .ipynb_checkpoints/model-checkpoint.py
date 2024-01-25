import torch.nn as nn
import torch.nn.functional as F
import torch

class Embedding(nn.Module): # token and positional embedding
    def __init__(self, vocab_size, n_embd, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout) # embedding dropout

    
    def forward(self, x): # -> returns (B, T, n_embd)
        tok = self.tok_emb(x)
        
        assert tok.size(1) <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        pos = self.pos_emb(torch.arange(tok.size(1), dtype=torch.long).unsqueeze(0)) #(1, T, C)
        
        x = tok + pos
        
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, fan_in_out, n_embd, bias=None):
        super().__init__()
        self.fan_in_out = fan_in_out
        # MLP assembler
        self.mlp = nn.Sequential(
            nn.Linear(fan_in_out, n_embd),
            nn.LayerNorm(n_embd),
            nn.GELU(),
            nn.Dropout(0.2), # mlp dropout, just prefer dropout between layers rather than at the end.
            nn.Linear(n_embd, fan_in_out),
        )
        
    def forward(self, x):
        assert x.size(-1) == self.fan_in_out
        return self.mlp(x)


# NOTE: this isn't the best way to implement attention!!!
class Attention(nn.Module):
    def __init__(self, config, mask=None):
        self.n_head = config.n_head
        self.affine = nn.Linear(config.n_embd, 3*config.n_embd, device=config.device)
        self.drop = config.dropout
        self.dropout1 = nn.Dropout(self.drop)
        self.dropout2 = nn.Dropout(self.drop)


        # https://github.com/karpathy/nanoGPT/blob/master/model.py#L45C1-L46C1
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, block_size, block_size).expand(4, -1, -1)
                             if mask is None else mask)

        self.out = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v  = self.affine(x).chunk(3, dim=2)
        # att = torch.zeros(B, T, T) # <- here

        assert C % self.n_head == 0, f"{self.n_head=} is supposed to be a multiple of {C=}"
        
        key = k.view(B, T, self.n_head, C//self.n_head).transpose(2, 1) # (B, n_head, T, head_dim) 
        query = q.view(B, T, self.n_head, C//self.n_head).transpose(2, 1) # (B, n_head, T,  head_dim)
        value = v.view(B, T, self.n_head, C//self.n_head).transpose(2, 1) # (B, n_head, T,  head_dim)

        if self.flash: # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.drop, is_causal=True)
        
        else:
            att = (query @ key.transpose(2, 3)) / math.sqrt(C//self.n_head)
            att = att.masked_fill(self.mask[:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
    
            att = self.dropout1(att)
            y = att @ value
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y= self.out(y)
        
        assert x.shape == y.shape
        return self.dropout2(y)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ma_ln = nn.LayerNorm(n_embd)
        self.masked_att = Attention()
        
        # MLP assembler
        self.mlp_ln = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, 4*n_embd)


        
    def forward(self, x):
        x = x + self.masked_att(self.ma_ln(x)) # seen in a lot of reference that normalization is done before layer rather than after skip connections.
        x = x + self.mlp(self.mlp_ln(x))  
        return x

# a decoder transformer model
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = block_size
        self.emb = Embedding(vocab_size, n_embd, dropout=dropout)

        self.block = nn.Sequential(DecoderBlock() for i in range(n_block))
        
        # MLP assembler
        self.ln = nn.LayerNorm(n_embd)

        self.out = nn.Linear(n_embd, vocab_size)


        
    def forward(self, x, targets=None):
        x = self.emb(x) #(B, T, n_embd)
        
        x = self.block(x) # seen in a lot of reference that normalization is done before layer rather than after skip connections.
        x = self.ln(x)
        x = self.out(x) # (B, T, vocab_size)
        
        B, T, C = x.shape
        loss = F.cross_entropy(x.view(B*T, C), targets.view(B*T), ignore_index=-1) if targets is not None else None
        
        return x, loss
