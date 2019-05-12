import unittest
import torch
from source.nn import ContextQueryAttention


class ContextQueryAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        in_ch = 7
        l1, l2 = 9, 10
        attn = ContextQueryAttention(in_ch)
        context = torch.rand((5, l2, in_ch))
        query = torch.rand((5, l1, in_ch))
        self.assertEqual(torch.Size([5, l2, in_ch*4]), attn(context, query).shape)
