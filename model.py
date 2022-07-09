import torch as th
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):

    def __init__(self,
                 t_max: int,
                 d_model: int):

        super(PositionalEncoding, self).__init__()
        pos, emb_dim = th.meshgrid(th.arange(t_max), th.arange(d_model), indexing='ij')
        self.pos_enc = pos / (10000 ** (emb_dim / d_model))
        self.pos_enc[:, 0::2] = th.sin(self.pos_enc[:, 0::2])
        self.pos_enc[:, 1::2] = th.cos(self.pos_enc[:, 1::2])

    def forward(self,
                mask: Optional[th.Tensor] = None):

        if mask is not None:
            pos_enc = self.pos_enc.masked_fill(mask == 0, 0)
        return pos_enc


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_k: int,
                 d_v: int):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v

        self.Wq = nn.Linear(in_features=d_model, out_features=d_k * heads, bias=False)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_k * heads, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_v * heads, bias=False)
        self.Wo = nn.Linear(in_features=d_v * heads, out_features=d_model, bias=False)

    def forward(self,
                queries: th.Tensor,
                keys: th.Tensor,
                values: th.Tensor,
                mask: Optional[th.Tensor] = None):
        """

        :param queries: The Q matrix of the multi-head attention mechanism with shape (n, t, d_model)
        :param keys: The K matrix of the multi-head attention mechanism with shape (n, t, d_model)
        :param values: The V matrix of the multi-head attention mechanism with shape (n, t, d_model)
        :param mask: The binary mask tensor, where 0 indices are ignored. The attention of these indices is set to 0.
        :return: The multi-head attention output with shape (n, t, d_model)
        """

        n, t = queries.shape[:1]

        # Compute the queries Qh, keys Kh and values Vh for each head and transpose the matrices to multiply them
        Qh = self.Wq(queries).view(n, t, self.heads, self.d_k).transpose(1, 2)
        Kh = self.Wk(keys).view(n, t, self.heads, self.d_k).transpose(1, 2)
        Vh = self.Wv(values).view(n, t, self.heads, self.d_v).transpose(1, 2)

        # Calculate the attention output for each head. Do not pay attention to the masked elements
        energy = th.matmul(Qh, Kh.transpose(1, 2)) / (self.d_k ** 1/2)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)
        attention = F.softmax(energy, dim=-1)
        head_i = th.matmul(attention, Vh)

        # Calculate the multi-head attention
        concat = head_i.transpose(1, 2).view(n, t, self.heads * self.d_v)
        out = self.Wo(concat)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 d_ff: int,
                 dropout: float
                 ) -> None:
        super(TransformerBlock, self).__init__()

        self.attention_block = MultiHeadAttention(heads, d_model, d_k, d_v)
        self.attention_norm = nn.LayerNorm(d_model)

        self.ff_block = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                queries: th.Tensor,
                keys: th.Tensor,
                values: th.Tensor,
                mask: Optional[th.Tensor] = None):

        attn_out = self.attention_norm(self.attention_block(queries, keys, values, mask) + queries)
        attn_out = self.dropout(attn_out)

        ff_out = self.ff_norm(self.ff_block(attn_out) + attn_out)
        ff_out = self.dropout(ff_out)

        return ff_out


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 N: int,
                 t_max: int,
                 heads: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 d_ff: int,
                 dropout: float
                 ) -> None:

        self.d_model = d_model

        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(t_max, d_model)
        self.encoder = nn.ModuleList([TransformerBlock(heads, d_model, d_k, d_v, d_ff, dropout) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: th.Tensor,
                mask: Optional[th.Tensor] = None):

        x = self.word_emb(x) + self.pos_enc(mask)
        x = self.dropout(x)
        for encoder_block in self.encoder:
            x = encoder_block(x, x, x, mask)
        return x

