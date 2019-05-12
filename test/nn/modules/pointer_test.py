import unittest
import torch
from source.nn import Pointer


class PointerTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size = 5
        l = 3
        in_ch, kernel_size = 5, 7
        pointer = Pointer(in_ch)
        m0 = torch.rand((batch_size, l, in_ch))
        m1 = torch.rand((batch_size, l, in_ch))
        m2 = torch.rand((batch_size, l, in_ch))
        p1, p2 = pointer(m0, m1, m2)
        self.assertEqual(torch.Size([batch_size, l]), p1.shape)
        self.assertEqual(torch.Size([batch_size, l]), p2.shape)
