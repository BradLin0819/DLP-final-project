import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, DistilBertModel, RobertaModel

class BertbiLSTM(nn.Module):
    def __init__(self, bert='bert-base-cased', n_layers=2, bidirectional=True, 
            output_dim=1, dropout=0.3, hidden_dim=None, 
            freeze=True):
        super(BertbiLSTM, self).__init__()

        self.bert = RobertaModel.from_pretrained(bert)
        self.freeze = freeze
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(embedding_dim, hidden_size=embedding_dim if hidden_dim is None else hidden_dim, 
                    num_layers=n_layers, bidirectional=bidirectional, batch_first=True, 
                    dropout=0 if n_layers < 2 else dropout)
        lstm_input_dim = embedding_dim if hidden_dim is None else hidden_dim
        self.fc = nn.Linear(2 * lstm_input_dim if bidirectional else lstm_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if freeze:
            self.freeze_bert()

    def forward(self, text):
        #text = [batch size, sent_length]
        embedded = None
        attention_mask = (text != self.bert.config.pad_token_id)
        if self.freeze:
            with torch.no_grad():
                embedded = self.bert(text, attention_mask=attention_mask)[0]
        else:
            embedded = self.bert(text, attention_mask=attention_mask)[0]
        _, (hidden, cell) = self.lstm(embedded)
        
        #embedded = [batch size, sent len, emb dim]
        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        return self.fc(hidden)
    
    def freeze_bert(self):
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False