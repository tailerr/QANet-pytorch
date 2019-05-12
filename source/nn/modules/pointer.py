import torch.nn as nn
import torch
import math
from .embedding import InitializedConv1d
from .attention import mask_logits


class Pointer(nn.Module):
    """
    Input:
        m0: a float tensor of shape (batch, length, model_dim)
        m1: a float tensor of shape (batch, length, model_dim)
        m2: a float tensor of shape (batch, length, model_dim)
    Output:
        p1: a float tensor of shape (batch, length)
        p2: a float tensor of shape (batch, length)
    """
    def __init__(self, model_dim):
        super(Pointer, self).__init__()
        self.w1 = InitializedConv1d(model_dim*2, 1)
        self.w2 = InitializedConv1d(model_dim*2, 1)

    def forward(self, m0, m1, m2, mask):
        p1 = torch.cat([m0, m1], 2).transpose(1, 2)
        p2 = torch.cat([m0, m2], 2).transpose(1, 2)
        p1 = mask_logits(self.w1(p1).squeeze(), mask)
        p2 = mask_logits(self.w2(p2).squeeze(), mask)
        p1 = nn.functional.log_softmax(p1, 1)
        p2 = nn.functional.log_softmax(p2, 1)
        return p1, p2
