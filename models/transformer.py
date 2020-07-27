import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import re

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, 
            nhid, nlayers, pretrained_vec=None, 
            dropout=0.2):
        super().__init__()

        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(int(ntoken), int(ninp))  #[input_size, hidden_size]
        self.ninp = ninp
        self.pretrained_vec = pretrained_vec
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        if self.pretrained_vec is not None:               
            self.embedding.from_pretrained(self.pretrained_vec)
        else:
            initrange = 0.1
            self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, pad_mask=None):
        src = self.embedding(src) * math.sqrt(self.ninp)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=pad_mask)
        # [L, batch, d_model]
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

if __name__ == '__main__':
    ntokens = 1e3
    embed = 300
    nhid = 256
    nhead = 4
    nlayers = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.randint(0, int(ntokens), (500, 8))
    
    model = TransformerModel(ntoken=ntokens, ninp=embed, nhid=nhid, nlayers=nlayers, nhead=nhead)
    print(model(data).size())
    
        
    

