from nn import *
from data import EMA


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat, emb_word_size, emb_char_size, model_dim, pretrained_char,
                 max_pass_len, max_ques_len, kernel_size, num_heads, block_num, ch_dropout, w_dropout,
                 grad_clip, ema_decay, length=400):
        super(QANet, self).__init__()
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.embedding = Embedding(emb_char_size, ch_dropout, w_dropout, emb_word_size, model_dim)
        self.context_conv = DepthwiseSeparatableConv(model_dim, model_dim, 5)
        self.question_conv = DepthwiseSeparatableConv(model_dim, model_dim, 5)
        self.c_emb_encoder = Encoder(model_dim, max_pass_len, kernel_size, num_heads, 4, length=400)
        self.q_emb_encoder = Encoder(model_dim, max_ques_len, kernel_size, num_heads, 4, length=50)
        self.c_q_attn = ContextQueryAttention(model_dim)
        self.cq_conv = DepthwiseSeparatableConv(4*model_dim, model_dim, 5)
        self.model_encoders = nn.ModuleList([Encoder(model_dim, max_pass_len, kernel_size, num_heads, 2, length=400)
                                            for _ in range(block_num)])
        self.pointer = Pointer(model_dim)
        self.criterion = nn.NLLLoss(reduction='mean')
        self.grad_clip = grad_clip
        self.ema = EMA(ema_decay)
        self.dropout = w_dropout
        self.PAD = 0
        for name, p in self.named_parameters():
            if p.requires_grad: self.ema.set(name, p)
        
    def forward(self, passage, passage_ch, question, question_ch):
        mask_p = (torch.ones_like(passage) * self.PAD != passage).float()
        mask_q = (torch.ones_like(question) * self.PAD != question).float()
        passage_w_emb, passage_c_emb = self.word_emb(passage), self.char_emb(passage_ch)
        question_w_emb, question_c_emb = self.word_emb(question), self.char_emb(question_ch)
        passage_emb = self.embedding(passage_c_emb, passage_w_emb)
        question_emb = self.embedding(question_c_emb, question_w_emb)
        c = self.context_conv(passage_emb)
        q = self.question_conv(question_emb)
        c = self.c_emb_encoder(c, mask(c))
        q = self.q_emb_encoder(q, mask(q))
        cq = self.c_q_attn(c, q, mask_p, mask_q)
        m1 = self.cq_conv(cq)
        m1 = nn.functional.dropout(m1, self.dropout, self.training)
        for enc_block in self.model_encoders:
            m1 = enc_block(m1)
        m2 = m1
        for enc_block in self.model_encoders:
            m2 = enc_block(m2)
        m3 = m2
        for enc_block in self.model_encoders:
            m3 = enc_block(m3)
        p1, p2 = self.pointer(m1, m2, m3, mask_p)
        return p1, p2
           
    def train_step(self, batch, y1, y2, optim, scheduler):
        self.train()
        optim.zero_grad()
        p1, p2 = self.forward(*batch)
        loss1 = self.criterion(p1, y1)
        loss2 = self.criterion(p2, y2)
        loss = (loss1+loss2)/2
        loss.backward()
        optim.step()
        scheduler.step()
        for name, p in self.named_parameters():
                if p.requires_grad:
                    self.ema.update_parameter(name, p)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        return loss.item(), p1, p2

    def evaluate(self, batch, y1, y2):
        self.eval()
        with torch.no_grad():
            p1, p2 = self.forward(*batch)
            loss1 = self.criterion(p1, y1)
            loss2 = self.criterion(p2, y2)
            loss = (loss1+loss2)/2
        return loss.item(), p1, p2


def mask(batch):
    batch_size, seq_len, _ = batch.shape
    x = torch.ones(seq_len, seq_len, device=batch.device).tril(-1).transpose(0, 1)

    return x.repeat(batch_size, 1, 1).byte()
