import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerModel

class TEbiLSTM(nn.Module):
    def __init__(self, ntoken, d_model=512, 
            nhead=8, nhid=512, te_nlayers=6, 
            pretrained_vec=None, n_layers=2, bidirectional=True, 
            output_dim=1, hidden_dim=256, dropout=0.3, 
            pad_token_id=None
            ):
        super(TEbiLSTM, self).__init__()

        self.transformer_encoder = TransformerModel(ntoken, d_model, nhead, 
                                                nhid, te_nlayers, pretrained_vec, 
                                                dropout)
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.lstm = nn.LSTM(d_model, hidden_size=hidden_dim, 
                    num_layers=n_layers, bidirectional=bidirectional, 
                    dropout=0 if n_layers < 2 else dropout)
        lstm_input_dim = d_model if hidden_dim is None else hidden_dim
        self.fc = nn.Linear(2 * lstm_input_dim if bidirectional else lstm_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _padding_mask(self, src, pad_token_id):
        mask = (src == pad_token_id)
        return mask

    def forward(self, text):
        #text = [batch size, sent_len]
        pad_mask = None

        if self.pad_token_id is not None:
            pad_mask = self._padding_mask(text, self.pad_token_id)

        #embedded = [sent len, batch size, emb dim]
        embedded = self.transformer_encoder(text, pad_mask=pad_mask)
        _, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        return self.fc(hidden)