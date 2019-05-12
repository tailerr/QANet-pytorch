from source.nn import DepthwiseSeparatableConv, MultiHeadAttention, EmbeddingEncoderLayer, PositionalEncoding
import unittest
import torch


class DepthwiseSeparatableConvTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size = 5
        l = 3
        in_ch, kernel_size = 5, 7
        conv = DepthwiseSeparatableConv(in_ch, in_ch, kernel_size)
        batch = torch.rand((batch_size, in_ch, l))
        self.assertEqual(batch.transpose(1, 2).shape, conv(batch).shape)


class MultiHeadAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size, l, l1 = 5, 6, 8
        dim_m = 7
        value = torch.rand((batch_size, l, dim_m))
        key = torch.rand((batch_size, l1, dim_m))
        attn = MultiHeadAttention(2, dim_m, dim_m, dim_m, 0.2)
        self.assertEqual(value.shape, attn(value, key).shape)


class PositionalEncodingTest(unittest.TestCase):
    def test_output_shape(self):
        model_dim, max_seq_len = 6, 10
        encoding = PositionalEncoding(model_dim, max_seq_len)
        batch = torch.rand((5, 4, model_dim))
        self.assertEqual(batch.shape, encoding(batch).shape)


class EmbeddingEncoderLayerTest(unittest.TestCase):
    def test_output_shape(self):
        model_dim, max_seq_len = 7, 10
        kernel_size, num_heads = 3, 2
        encoder = EmbeddingEncoderLayer(model_dim, max_seq_len, model_dim, model_dim, kernel_size, num_heads, model_dim,
                                        model_dim, 0.2)
        batch = torch.rand((5, 8, model_dim))
        self.assertEqual(batch.shape, encoder(batch).shape)

