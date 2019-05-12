import torch.nn as nn
import torch
import numpy as np
from torch.nn.init import kaiming_normal_
from .embedding import InitializedConv1d


class Encoder(nn.Module):
    """
    Args:
        described in internal classes
    Input:
        batch: a float tensor of shape (batch, length, model_dim)
    Output:
        batch: a float tensor of shape (batch, length, model_dim)
    """
    def __init__(self, model_dim, max_seq_len, kernel_size, num_heads, conv_number, length, in_ch=None, out_ch=None,
                 dim_q_k=None, dim_v=None, dropout=0.1):
        super(Encoder, self).__init__()
        if dim_q_k is None:
            dim_q_k = model_dim
            dim_v = model_dim
        if in_ch is None:
            in_ch = model_dim
            out_ch = model_dim
        self.position_encoding = PositionalEncoding(model_dim, max_seq_len)
        self.norm_2 = nn.LayerNorm([length, model_dim])
        self.norm_1 = nn.LayerNorm([length, model_dim])
        self.norms = nn.ModuleList([nn.LayerNorm([length, model_dim], eps=1e-12) for _ in range(conv_number)])
        self.convs = nn.ModuleList([DepthwiseSeparatableConv(in_ch, out_ch, kernel_size) for _ in range(conv_number)])
        self.attn = MultiHeadAttention(num_heads, model_dim, dim_q_k, dim_v, dropout)
        self.conv_1 = InitializedConv1d(model_dim, model_dim, relu=True, bias=True)
        self.conv_2 = InitializedConv1d(model_dim, model_dim, bias=True)
        self.conv_number = conv_number
        self.dropout = dropout

    def forward(self, batch, mask=None):
        batch = self.position_encoding(batch)
        for i, conv in enumerate(self.convs):
            residual = batch
            batch = self.norms[i](batch)
            if i % 2 == 0:
                batch = nn.functional.dropout(batch, p=self.dropout, training=self.training)
            batch = conv(batch)
            batch = batch + residual
        residual = batch
        batch = self.norm_1(batch)
        batch = nn.functional.dropout(batch, self.dropout, self.training)
        batch = self.attn(batch, mask=mask)+residual
        batch = nn.functional.dropout(batch, p=self.dropout, training=self.training)
        residual = batch

        batch = self.norm_2(batch)
        batch = nn.functional.dropout(batch, p=self.dropout, training=self.training)
        batch = self.conv_1(batch.transpose(1, 2))
        batch = self.conv_2(batch).transpose(1, 2)
        batch = batch + residual
        return nn.functional.dropout(batch, p=self.dropout, training=self.training)


class PositionalEncoding(nn.Module):
    """
    Input:
        batch: a float tensor of shape (batch, length, model_dim)
    """
    def __init__(self, model_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dim = model_dim
        self.max_seq_len = max_seq_len
        self.positional = nn.Parameter(torch.tensor(self.set_weights(), dtype=torch.float), requires_grad=False)

    def set_weights(self):
        w = [pos/np.power(1e4, np.arange(0, self.dim)/self.dim) for pos in range(self.max_seq_len)]
        w = np.stack(w)
        w[1:, ::2] = np.sin(w[1:, ::2])
        w[1:, 1::2] = np.cos(w[1:, 1::2])
        return w

    def forward(self, input):
        return input + self.positional[0:input.size(1)]


class DepthwiseSeparatableConv(nn.Module):
    """
        Args:
            in_ch: Number of channels in the input (emb_size)
            kernel_size (int or tuple): Size of the convolving kernel
        Input:
              **x** of shape `(batch, input_size, length)`:  a float tensor
        Output:
                a float tensor of shape (batch, length, output_size)
    """
    def __init__(self, in_ch, out_ch, kernel_size):
        super(DepthwiseSeparatableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_ch, in_ch, kernel_size, groups=in_ch,
                                        padding=kernel_size//2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_ch, out_ch, 1, padding=0)
        self.in_ch = in_ch
    
    def forward(self, x):
        if x.size(1) != self.in_ch:
            x.transpose_(1, 2)
        k = self.pointwise_conv(self.depthwise_conv(x)).transpose(1, 2)
        return nn.functional.relu(k)


class ScaledDotProductAttention(nn.Module):
    """
    Input:
        value: float tensor of shape (batch*num_heads, seq_len, dim_m)
        query: float tensor of shape (batch*num_heads, seq_len, dim_m)
        key: float tensor of shape (batch*num_heads, q_len, dim_m)
        mask: float tensor of shape (batch*num_heads, q_len, seq_len)
    Output:
        attention: a float tensor of shape (batch*num_heads, length, output_size)
    """
    def __init__(self, input_size):
        super(ScaledDotProductAttention, self).__init__()
        self.input_size = input_size
        self.scale_factor = np.power(input_size, -0.5)

    def forward(self, value, query, key, mask=None):
        outputs = torch.bmm(value, key.permute([0, 2, 1]))*self.scale_factor
        if mask is not None:
            outputs.masked_fill_(mask, -float('inf'))
        attention = nn.functional.softmax(outputs, 2)
        attention = torch.bmm(attention, key)
        return attention


class MultiHeadAttention(nn.Module):
    """
        Args:
            num_heads: heads number in attention
            dim_m: model dimention
            dim_q_k: query and key dimention
            dim_v: value dimention
        Input:
            value: float tensor of shape (batch, seq_len, dim_m)
            query: float tensor of shape (batch, seq_len, dim_m)
            key: float tensor of shape (batch, q_len, dim_m)
            mask: float tensor of shape (batch, q_len, seq_len)
        Output:
            attention: float tensor of shape (batch, q_len, dim_m)
    """
    def __init__(self, num_heads, dim_m, dim_q_k, dim_v, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = num_heads
        self.model_dim = dim_m
        self.dim_q_k = dim_q_k
        self.dim_v = dim_v

        self.dropout = nn.Dropout(dropout)
        self.query_projection = nn.Parameter(torch.FloatTensor(num_heads, dim_m, dim_q_k))
        self.key_projection = nn.Parameter(torch.FloatTensor(num_heads, dim_m, dim_q_k))
        self.value_projection = nn.Parameter(torch.FloatTensor(num_heads, dim_m, dim_v))
        self.attention = ScaledDotProductAttention(dim_q_k)
        self.final_projection = nn.Linear(num_heads*dim_v, dim_m)
        for parameter in [self.query_projection, self.key_projection, self.value_projection]:
            kaiming_normal_(parameter.data)

    def forward(self, value, key=None, query=None, mask=None):
        if query is None:
            query = value
        if key is None:
            key = value
        seq_len = value.shape[1]
        q_len = key.shape[1]
        batch_size = query.shape[0]
        value, query, key = map(self.stack_heads, [value, query, key])
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
        # (n_heads, batch * x, dim_m) -> (n_heads, batch * x, projection) -> (n_heads * batch, x, projection)
        value = value.bmm(self.value_projection).view(-1, seq_len, self.dim_v)
        key = key.bmm(self.key_projection).view(-1, q_len, self.dim_q_k)
        query = query.bmm(self.query_projection).view(-1, seq_len, self.dim_q_k)

        context = self.attention(value, query, key, mask)

        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        output = self.final_projection(concat_heads)
        output = self.dropout(output)

        return output

    def stack_heads(self, tensor):
        return tensor.view(-1, self.model_dim).repeat(self.n_heads, 1, 1)
