import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)].cuda(),
                         requires_grad=False)
        #x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t))).unsqueeze(0)



class NewRelTemporalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=20):
        super(NewRelTemporalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


        # proposals
        self.num_clips = 32
        start = np.reshape(np.repeat(np.arange(0, self.num_clips)[:, np.newaxis], axis=1,
                                     repeats=self.num_clips), [-1])
        end = np.reshape(np.repeat(np.arange(1, self.num_clips + 1)[np.newaxis, :], axis=0,
                                   repeats=self.num_clips), [-1])
        self.props = np.stack([start, end], -1)

        # predefined proposals
        idx = self.props[:, 0] < self.props[:, 1]
        self.idx = idx
        self.props = self.props[idx]
        self.props = torch.from_numpy(self.props).cuda()
    def forward(self, x):
        bsz = x.size(0)
        left_pos = self.props[:, 0]#.expand(x.size(1), -1)
        right_pos = self.props[:, 1]#.expand(x.size(1), -1)
        pos_emb = Variable(self.pe[:, :x.size(1)].cuda(),
                         requires_grad=False) # [1, 528, 256]
        left_pos = left_pos.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 256)
        right_pos = right_pos.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 256)
        pos_emb_l = pos_emb.gather(dim=1, index=left_pos)
        pos_emb_r = pos_emb.gather(dim=1, index=right_pos)
        pos_emb_cat = torch.cat([pos_emb_l, pos_emb_l], -1)
        x = x + pos_emb_cat
        # x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

if __name__ == '__main__':
    rel_pos = RelTemporalEncoding(n_hid=512, max_len=240).cuda()
    c = torch.LongTensor([1, 2, 3, 239])
    x = torch.rand([4, 512]).cuda()
    x = rel_pos(x, c)
    print("done.")
