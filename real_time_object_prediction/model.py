import torch as th
import torch.nn.functional as F
import torch.nn as nn
import warnings
from typing import Optional, Literal


class LearnedPositionalEncoding(nn.Module):

    def __init__(self,
                 max_pad_length: int,
                 d_model: int) -> None:
        super(LearnedPositionalEncoding, self).__init__()
        pos, emb_dim = th.meshgrid(th.arange(max_pad_length), th.arange(d_model), indexing='ij')
        pos_enc = pos / (10000 ** (emb_dim / d_model))  # (max_pad_length, d_model)
        pos_enc[:, 0::2] = th.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = th.cos(pos_enc[:, 1::2])

        self.pos_enc = nn.Parameter(pos_enc)

    def forward(self,
                mask: Optional[th.Tensor] = None):

        if mask is not None:
            mask = mask[..., None]
            pos_enc = self.pos_enc.masked_fill(mask == 0, 0)
        else:
            pos_enc = self.pos_enc
        return pos_enc


class PositionalEncoding(nn.Module):

    def __init__(self,
                 max_pad_length: int,
                 d_model: int) -> None:

        super(PositionalEncoding, self).__init__()
        pos, emb_dim = th.meshgrid(th.arange(max_pad_length), th.arange(d_model), indexing='ij')
        pos_enc = pos / (10000 ** (emb_dim / d_model))  # (max_pad_length, d_model)
        pos_enc[:, 0::2] = th.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = th.cos(pos_enc[:, 1::2])
        self.register_buffer('pos_enc', pos_enc[None, ...])  # (1, max_pad_length, d_model)

    def forward(self,
                mask: Optional[th.Tensor] = None):

        if mask is not None:
            mask = mask[..., None]
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
        Qh = self.Wq(queries).view(n, t, self.heads, self.d_head).transpose(2, 3)  # (n, t, d_head, heads)
        Kh = self.Wk(keys).view(n, t, self.heads, self.d_head)  # (n, t, heads, d_head)
        Vh = self.Wv(values).view(n, t, self.heads, self.d_head).transpose(2, 3)  # (n, t, d_head, heads)

        # Calculate the attention output for each head. Do not pay attention to the masked elements
        energy = th.matmul(Qh, Kh) / (self.d_head ** 1 / 2)  # (n, t, d_head, d_head)
        if mask is not None:
            mask = mask[..., None, None]
            energy = energy.masked_fill(mask == 0, -1e20)
        attention = F.softmax(energy, dim=-1)  # (n, t, d_head, d_head)
        head_i = th.matmul(attention, Vh)  # (n, t, d_head, heads)

        # Calculate the multi-head attention
        concat = head_i.view(n, t, self.d_model)  # (n, t, d_model)
        out = self.Wo(concat)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float,
                 activation: Literal['relu', 'gelu'],
                 norm_mode: Literal['batch_norm', 'layer_norm']
                 ) -> None:
        super(TransformerBlock, self).__init__()

        self.norm_mode = norm_mode

        self.attention_block = MultiHeadAttention(heads, d_model)
        self.attention_norm = nn.BatchNorm1d(d_model) if norm_mode == 'batch_norm' else nn.LayerNorm(d_model)

        self.ff_block = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
        self.ff_norm = nn.BatchNorm1d(d_model) if norm_mode == 'batch_norm' else nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                queries: th.Tensor,
                keys: th.Tensor,
                values: th.Tensor,
                mask: Optional[th.Tensor] = None):
        attn_out = self.dropout(self.attention_block(queries, keys, values, mask)) + queries
        if self.norm_mode == 'batch_norm':
            attn_out = self.attention_norm(attn_out.transpose(1, 2)).transpose(1, 2)
        else:
            attn_out = self.attention_norm(attn_out)

        ff_out = self.dropout(self.ff_block(attn_out)) + attn_out
        if self.norm_mode == 'batch_norm':
            ff_out = self.ff_norm(ff_out.transpose(1, 2)).transpose(1, 2)
        else:
            ff_out = self.ff_norm(ff_out)

        return ff_out


class InputEncoding(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 input_enc_mode: Literal['conv_1d', 'linear'],
                 kernel_size: Optional[int] = None,
                 ):
        super(InputEncoding, self).__init__()

        self.d_model = d_model
        self.mode = input_enc_mode
        if input_enc_mode == 'linear':
            self.input_encoder = nn.Linear(d_input, d_model)
        else:

            if kernel_size % 2 == 1:
                pad = (kernel_size - 1) // 2
            else:
                lpad = (kernel_size - 1) // 2
                rpad = kernel_size - 1 - lpad
                pad = (lpad, rpad)

            self.input_encoder = nn.Conv1d(d_input, d_model, kernel_size, padding=pad)

    def forward(self, x):
        if self.mode == 'linear':
            x = self.input_encoder(x)
        else:
            x = self.input_encoder(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.d_model ** 0.5
        return x


class Encoder(nn.Module):
    def __init__(self,
                 N: int,
                 heads: int,
                 d_input: int,
                 max_pad_length: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float,
                 activation: Literal['relu', 'gelu'],
                 pos_encoder_mode: Literal['sinusoidal', 'learned'],
                 norm_mode: Literal['batch_norm', 'layer_norm'],
                 input_enc_mode: Literal['conv_1d', 'linear'],
                 kernel_size: Optional[int] = None
                 ) -> None:
        super(Encoder, self).__init__()

        self.input_enc = InputEncoding(d_input, d_model, input_enc_mode, kernel_size)
        if pos_encoder_mode == 'sinusoidal':
            self.pos_enc = PositionalEncoding(max_pad_length, d_model)
        else:
            self.pos_enc = LearnedPositionalEncoding(max_pad_length, d_model)
        self.transformer_enc = nn.ModuleList([TransformerBlock(heads, d_model, d_ff, dropout, activation, norm_mode)
                                              for _ in range(N)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: th.Tensor,
                pad_mask: Optional[th.Tensor] = None):
        x = self.input_enc(x) + self.pos_enc(pad_mask)
        x = self.dropout(x)
        for encoder_block in self.transformer_enc:
            x = encoder_block(x, x, x, pad_mask)
        return x


class ClassificationNet(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_pad_length: int,
                 d_output: int,
                 ) -> None:

        super(ClassificationNet, self).__init__()


        self.output_layer = nn.Linear(d_model * max_pad_length, d_output)

    def forward(self, x):
        x = self.conv_layer(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = x.reshape(x.shape[0], -1)
        x = self.output_layer(x)
        return x


class GraspMovNet(nn.Module):
    def __init__(self,
                 N: int,
                 heads: int,
                 d_input: int,
                 d_output: int,
                 max_pad_length: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float,
                 activation: Literal['relu', 'gelu'],
                 mode: Literal['pretrain', 'train'],
                 pos_encoder_mode: Literal['sinusoidal', 'learned'],
                 norm_mode: Literal['batch_norm', 'layer_norm'],
                 input_enc_mode: Literal['conv_1d', 'linear'],
                 kernel_size: Optional[int] = None
                 ) -> None:
        super(GraspMovNet, self).__init__()

        assert mode in ['pretrain', 'train'], "The model can be either trained ('train') or pretrained ('pretrain')."

        assert pos_encoder_mode in ['sinusoidal', 'learned'], "The positional encoder can either be 'sinusoidal' " \
                                                              "or 'learned'."

        assert norm_mode in ['batch_norm', 'layer_norm'], "The linear combination of the neurons' inputs are " \
                                                          "normalized either using 'batch_norm' or 'layer_norm'."

        assert input_enc_mode in ['conv_1d', 'linear'], "The input is encoded using either a 'conv_1d' or a 'linear' " \
                                                        "layer."

        if input_enc_mode == 'conv_1d':
            assert kernel_size is not None, "The kernel size must be specified only if the input is encoded using a " \
                                            "1d Convolution."
        else:
            assert kernel_size is None, "The kernel size must be specified only if the input is encoded using a 1d " \
                                        "Convolution."

        assert activation in ['relu', 'gelu'], "The activation function is either 'relu' or 'gelu'."

        self.mode = mode
        self.max_pad_length = max_pad_length
        self.d_model = d_model
        self.d_input = d_input
        self.d_output = d_output
        self.kernel_size = kernel_size
        self.activation = activation

        self.encoder_net = Encoder(N, heads, d_input, max_pad_length, d_model, d_ff, dropout, activation,
                                   pos_encoder_mode, norm_mode, input_enc_mode, kernel_size)

        if self.mode == 'pretrain':
            self.output_net = nn.Linear(d_model, d_input)
        else:
            self.output_net = nn.Linear(d_model * max_pad_length, d_output)

    def set_mode(self,
                 mode: Literal['pretrain', 'train']
                 ) -> None:

        assert mode in ['pretrain', 'train'], "The model can be either trained ('train') or pretrained ('pretrain')."

        if self.mode == mode:
            warnings.warn(f"The network is already used for the '{self.mode}' task. The output layer will not be "
                          f"reinitialized.")
        else:
            self.mode = mode
            if mode == 'pretrain':
                self.output_net = nn.Linear(self.d_model, self.d_input)
            else:
                self.output_net = nn.Linear(self.d_model * self.max_pad_length, self.d_output)

    def forward(self,
                x: th.Tensor,
                pad_mask: Optional[th.Tensor] = None) -> th.Tensor:

        x = self.encoder_net(x, pad_mask)
        if self.mode == 'train':
            x = x.reshape(x.shape[0], -1)
        x = self.output_net(x)
        return x
