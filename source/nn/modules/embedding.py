import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, char_emb_size, ch_dropout, w_dropout, word_emb_size, model_dim):
        super(Embedding, self).__init__()
        self.ch_dropout = ch_dropout
        self.w_dropout = w_dropout
        self.highway = Highway(2, model_dim, w_dropout)
        self.ch_conv = nn.Conv2d(char_emb_size, model_dim, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.ch_conv.weight, nonlinearity='relu')
        self.conv = InitializedConv1d(word_emb_size+model_dim, model_dim, bias=False)

    def forward(self, ch_emb, w_emb):
        # ch permute 0 3 1 2
        ch_emb = ch_emb.permute([0, 3, 1, 2])
        nn.functional.dropout(ch_emb, self.ch_dropout, self.training, True)
        nn.functional.dropout(w_emb, self.w_dropout, self.training, True)
        ch_emb = self.ch_conv(ch_emb)
        ch_emb = nn.functional.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3) #dim 3 2
        w_emb = w_emb.transpose(1, 2)
        emb = torch.cat((w_emb, ch_emb), 1) # dim 1 2
        emb = self.conv(emb)
        return self.highway(emb)
        

class Highway(nn.Module):
    def __init__(self, num_layers, size, dropout):
        super(Highway, self).__init__()
        self.n = num_layers
        self.dropout = dropout
        self.linear = nn.ModuleList([InitializedConv1d(size, size, relu=False, bias=True) 
                                     for _ in range(num_layers)])
        self.gate = nn.ModuleList([InitializedConv1d(size, size, bias=True) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = nn.functional.dropout(nonlinear, self.dropout, self.training)
            x = gate * nonlinear + (1-gate)*x
        return x


class InitializedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=padding,
                              groups=groups, bias=bias)
        if relu:
            self.relu = True
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        if self.relu:
            return nn.functional.relu(self.conv(x))
        return self.conv(x)         
