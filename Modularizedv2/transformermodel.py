import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention (nn.Module):
    def __init__(
        self,
        embedding_dim,
        heads
    ):

        super(SelfAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim//heads

        assert (self.head_dim * self.heads ==
                self.embedding_dim), 'embed size must be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # after concatenation of heads
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embedding_dim)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]

        # Lengths will mostly be sequence length for targets or src
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into number of heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # query shape(n, query_len, heads, head_dim)
        # keys shape(n, key_len, heads, head_dim)
        # energy shape (n, heads, query_len, key_len)
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # Normalize across sentence(dim=3)
        attention = torch.softmax(energy/(self.embedding_dim**(1/2)), dim=3)

        # attention shape (N, heads, query_len, key_len)
        # after einsum (N, query_len, heads, head_dim)
        # out shape (N, query_len, embedding_dim or head_dim * heads)

        # print(
        #     f'attention: {attention.shape} values: {values.shape} value: {value_len} key: {key_len} query: {query_len} head_dim: {self.head_dim}')
        out = (
            torch.einsum(
                'nhql,nlhd->nqhd', [attention, values])
            .reshape(N, query_len, self.heads*self.head_dim)
        )

        out = self.fc_out(out)

        return out


class TransformerBlock (nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        forward_expansion=4,
        dropout=0.0
    ):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim*forward_expansion, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attended = self.attention(values, keys, queries, mask)

        # shouldnt queries+attention be larger than embedding_dim
        x = self.dropout(self.norm1(attended + queries))
        fed_forward = self.ff(x)
        out = self.dropout(self.norm2(fed_forward + x))  # too many dims?

        return out


class TransformerNet (nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):

        super(TransformerNet, self).__init__()

        self.embedding_dim = args.embedding_dim
        self.device = kwargs['device']

        self.word_embedding = nn.Embedding(
            kwargs['vocab_size'], self.embedding_dim)
        self.pos_embedding = nn.Embedding(
            kwargs['max_len'], self.embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.embedding_dim,
                    args.num_heads,
                    args.forward_expansion,
                    args.dropout_rate
                ) for _ in range(args.num_layers)
            ]
        )

        self.dropout = nn.Dropout(args.dropout_rate)

        self.fc_out = nn.Linear(self.embedding_dim, 1)

    def forward(self, x, mask=None):
        N, seq_len = x.shape

        # Positional encoding
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(
            x) + self.pos_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = out.mean(dim=1)
        out = self.fc_out(out)

        return F.log_softmax(out.squeeze(dim=1))

        # Args:
        # num heads
        # embedding_dim
        #
        # kwargs:
        # device
        # vocab_size
        # max_len


class sample_args:
    def __init__(self):
        self.num_heads = 8
        self.embedding_dim = 512
        self.num_layers = 4
        self.forward_expansion = 4
        self.dropout_rate = 0.4


if __name__ == '__main__':
    args = sample_args()
    device = torch.device('cpu')

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [
        1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    model = TransformerNet(args, device=device,
                           vocab_size=10, max_len=9).to(device)

    out = model(x)
    print(out)
    print(out.shape)
