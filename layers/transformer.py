import torch
import torch.nn as nn
import torch.nn.functional as F

def create_staircase_matrix(rows, cols, granularity):
    """
    Create a matrix with a staircase-like pattern of non-zero values.

    :param rows: Number of rows in the matrix
    :param cols: Number of columns in the matrix
    :param granularity: block size of the staircase
    :return: Torch tensor representing the staircase matrix
    """
    # Initialize a zero matrix
    matrix = torch.tril(torch.ones(rows, cols))

    # Fill the diagonal with ones
    diag_length = min(rows, cols)

    for i in range(0, diag_length, granularity):
        matrix[i:i+granularity, i:i+granularity] = 1

    return matrix


# hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 1600 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
dropout = 0.2
# vocab_size = 256

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, granularity, matrix_type="full"):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.matrix_type = matrix_type
        if self.matrix_type == "triangular":
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        elif self.matrix_type == "staircase":
            self.register_buffer('tril', create_staircase_matrix(block_size, block_size, granularity))

        self.dropout = nn.Dropout(dropout)
        self.matrix_type = matrix_type

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if self.matrix_type != "full":
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, granularity, matrix_type):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, granularity, matrix_type) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, granularity, matrix_type):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, granularity, matrix_type)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class AttentionUnit(nn.Module):

    def __init__(self, n_embd, n_head, block_size, n_layer, output_size, granularity, matrix_type):
        """
        n_embd: the size of feature channels
        n_head: the number of heads we'd like
        block_size: the maximum context size
        n_layer: the number of attention layers
        output_size: the number of output channels
        """
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, granularity=granularity, matrix_type=matrix_type) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, output_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T, _ = x.shape

        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # for _ in range(max_new_tokens):
        #     # crop idx to the last block_size tokens
        #     idx_cond = idx[:, -block_size:]
        #     # get the predictions
        #     logits, loss = self(idx_cond)
        #     # focus only on the last time step
        #     logits = logits[:, -1, :] # becomes (B, C)
        #     # apply softmax to get probabilities
        #     probs = F.softmax(logits, dim=-1) # (B, C)
        #     # sample from the distribution
        #     idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        #     # append sampled index to the running sequence
        #     idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        # return idx

# model = AttentionUnit(51200, 6, 6, 6, 51200)
# # random tensor of shape (bs, seq_len, n_embd)
# x = torch.randn(1, 6, 51200)
# print(model(x)[0].shape)
