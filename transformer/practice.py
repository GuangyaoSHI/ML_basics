import torch 
import torch.nn as nn
from torch.nn import functional as F



# define input embedding and positional embedding

# define head
n_embd = 100
head_size = 100
block_size = 1000 # context size


class Head(nn.module):
    """
    one head of self-attention
    """
    def __init__(self, head_size):
        super.__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.mask = torch.tril(torch.ones(block_size, block_size))
        
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        q = self.query(x) # (batch, time-step, head_size)
        k = self.key(x) # (batch, time-step, head_size)
        v = self.value(x) # (batch, time-step, head_size)
        
        # compute score
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #(batch, T, T)
        wei = wei.mask_fill(self.mask[:T, :T]==0, float("inf")) # (batch, T, T)
        wei = F.softmax(wei, dim=1)
        
        # compute output vector
        out = wei @ v # (B, T, T) @ (B, T, head size) -> (B, T, head size)
        return out
    
    
class MultiHeadAttention(nn.module):
    """
     multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super.__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=1)
        out = self.proj(out)
        return out

    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
            )
        
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self, n_embd, n_head):
        super.__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

        
        