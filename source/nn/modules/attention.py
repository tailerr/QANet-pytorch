import torch.nn as nn
import torch
import math


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target*mask + (1-mask)*(-1e30)


class ContextQueryAttention(nn.Module):
    """
    Input:
        context: a float tensor of shape (batch, l1, model_dim)
        query: a float tensor of shape (batch, l2, model_dim)
    Output:
        attention: a float tensor of shape (batch, l1, 4*model_dim)
    """
    def __init__(self, input_size, dropout=0.1):
        super(ContextQueryAttention, self).__init__()
        w = torch.empty(input_size * 3)
        lim = 1 / input_size
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)
        self.dropout = dropout

    def forward(self, context, query, mask_p, mask_q):
        c_len = context.size(1)
        q_len = query.size(1)
        nn.functional.dropout(context, self.dropout, self.training, True)
        nn.functional.dropout(query, self.dropout, self.training, True)
        c = context.repeat(q_len, 1, 1, 1).permute([1, 0, 2, 3])
        q = query.repeat(c_len, 1, 1, 1).permute([1, 2, 0, 3])
        cq = c*q
        s = torch.matmul(torch.cat((q, c, cq), 3), self.w).transpose(1, 2)
        mask_p = mask_p.view(context.size(0), c_len, 1)
        mask_q = mask_q.view(context.size(0), 1, q_len)
        s1 = nn.functional.softmax(mask_logits(s, mask_q), 1)
        s2 = nn.functional.softmax(mask_logits(s, mask_p), 1)
        
        a = torch.bmm(s1, query)
        l = torch.bmm(s1, s2.transpose(1, 2))
        b = torch.bmm(l, context)
        output = torch.cat((context, a, context*a, context*b), dim=2)
        return nn.functional.dropout(output, p=self.dropout)
