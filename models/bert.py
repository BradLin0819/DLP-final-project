from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np

class BertClassifier(nn.Module):
    def __init__(self, output_dim=1, config='bert-base-cased'):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config)
        self.dropout = nn.Dropout(p=0.25)
        self.classify = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.freeze_bert()

    def forward(self, input_ids):
        attention_mask = (input_ids != self.bert.config.pad_token_id)
        _, pooled_output = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                        )
        output = self.dropout(pooled_output)
        return self.classify(output)

    def freeze_bert(self):
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False