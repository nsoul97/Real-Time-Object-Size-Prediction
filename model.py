import torch as th
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):

    def __init__(self,
                 max_pad_length: int,
                 d_model: int):

        super(PositionalEncoding, self).__init__()
        pos, emb_dim = th.meshgrid(th.arange(max_pad_length), th.arange(d_model))
        self.pos_enc = pos / (10000 ** (emb_dim / d_model))
        self.pos_enc[:, 0::2] = th.sin(self.pos_enc[:, 0::2])
        self.pos_enc[:, 1::2] = th.cos(self.pos_enc[:, 1::2])

    def forward(self,
                mask: Optional[th.Tensor] = None):

        if mask is not None:
            pos_enc = self.pos_enc.masked_fill(mask == 0, 0)
        else:
            pos_enc = self.pos_enc
        return pos_enc


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 heads: int,
                 d_model: int):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.d_model = d_model

        assert d_model % heads == 0, "The model dimension 'd_model' must be divisible by the number of heads 'heads'."
        self.d_head = d_model // heads

        self.Wq = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wo = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

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

        n, t = queries.shape[:2]

        # Compute the queries Qh, keys Kh and values Vh for each head and transpose the matrices to multiply them
        Qh = self.Wq(queries).view(n, t, self.heads, self.d_head).transpose(2, 3)   # (n, t, d_head, heads)
        Kh = self.Wk(keys).view(n, t, self.heads, self.d_head)                      # (n, t, heads, d_head)
        Vh = self.Wv(values).view(n, t, self.heads, self.d_head).transpose(2, 3)    # (n, t, d_head, heads)

        # Calculate the attention output for each head. Do not pay attention to the masked elements
        energy = th.matmul(Qh, Kh) / (self.d_head ** 1/2)                           # (n, t, d_head, d_head)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)
        attention = F.softmax(energy, dim=-1)                                       # (n, t, d_head, d_head)
        head_i = th.matmul(attention, Vh)                                           # (n, t, d_head, heads)

        # Calculate the multi-head attention
        concat = head_i.view(n, t, self.d_model)                          # (n, t, d_model)
        out = self.Wo(concat)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float
                 ) -> None:
        super(TransformerBlock, self).__init__()

        self.attention_block = MultiHeadAttention(heads, d_model)
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


class InputEncoding(nn.Module):
    def __init__(self, d_input, d_model):
        super(InputEncoding, self).__init__()
        self.input_encoder = nn.Linear(d_input, d_model)

    def forward(self, x):
        return self.input_encoder(x)


class Encoder(nn.Module):
    def __init__(self,
                 N: int,
                 heads: int,
                 d_input: int,
                 max_pad_length: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float
                 ) -> None:
        super(Encoder, self).__init__()

        self.input_enc = InputEncoding(d_input, d_model)
        self.pos_enc = PositionalEncoding(max_pad_length, d_model)
        self.transformer_enc = nn.ModuleList([TransformerBlock(heads, d_model, d_ff, dropout) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: th.Tensor,
                mask: Optional[th.Tensor] = None):

        x = self.input_enc(x) + self.pos_enc(mask)
        x = self.dropout(x)
        for encoder_block in self.transformer_enc:
            x = encoder_block(x, x, x, mask)
        return x

