import pandas as pd
import nltk
import re
import sys
import torch
import torchtext
from torchtext import data
from nltk.corpus import stopwords
from nltk import TweetTokenizer
import numpy as np
from numerical import num_range


def clean_text(sent):
    eyes = "[8:=;]"
    nose = "['`\-]?"
    
    sent = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', r'', sent)
    sent = re.sub(r'/', ' / ', sent)
    sent = re.sub(r'@\w+', '<USER>', sent)
    sent = re.sub(rf'{eyes}{nose}[)d]+|[)d]+{nose}{eyes}', '<SMILE>', sent, flags=re.IGNORECASE)
    sent = re.sub(rf'{eyes}{nose}p+', '<LOLFACE>', sent, flags=re.IGNORECASE)
    sent = re.sub(rf'{eyes}{nose}\(+|\)+{nose}{eyes}', '<SADFACE>', sent)
    sent = re.sub(rf'{eyes}{nose}[\/|l*]', '<NEUTRALFACE>', sent)
    sent = re.sub(r'<3', '<HEART>', sent)
    sent = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '<NUMBER>', sent)
    sent = re.sub(r'#(\S+)', r'<HASTAG> \1', sent)
    sent = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', sent)
    sent = re.sub(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <ELONG>', sent)
    # sent = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', r'', sent)
    sent = re.sub(r'&amp;?', r'and', sent)
    sent = re.sub(r'&lt;', r'<', sent)
    sent = re.sub(r'&gt;', r'>', sent)
    sent = re.sub(r'\$([A-Za-z]+)', r'\1', sent)
    # sent = re.sub(r'([\S]+)-([\S]+)', r'\1 \2', sent)
    # sent = re.sub(r'([+-])(\d+\.?\d+)', r'\1 \2', sent)
    sent = re.sub(r'(\d+\.?\d+)[+-]', r'<NUMBER>', sent)
    # sent = re.sub(r'(\d+)/(\d+)', r'\1 / \2', sent)
    sent = re.sub(r'([+-])(\d{1,3}(,\d{3})*(\.\d+)?)', r'<NUMBER>', sent)
    # sent = re.sub(r"\b([^a-z0-9()<>'`\-]){2,}\b", lambda x: f"{x.group().lower()} <ALLCAPS>", sent)
    return sent.lower()

# tokenizer = TweetTokenizer()
# nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

def type_convert(token):
    new_token = token
    if re.match(r'^\d+\.?\d+$', token):
        new_token = str(num_range(float(token)))
    elif re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', token):
        new_token = ''.join(token.split(','))
        new_token = str(num_range(float(new_token)))

    return new_token

class Tokenize:
    def __init__(self, tokenizer=TweetTokenizer(), max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, sent):
        # res = []
        tokens = self.tokenizer.tokenize(clean_text(sent))
        # idx = 0
        tokens = [token for token in tokens if token not in stopwords_set]
        # while idx < len(tokens):
        #     if tokens[idx] not in stopwords_set:
        #         if tokens[idx] == '$' and (idx+1) < len(tokens) and re.match(r'[^a-zA-Z]', tokens[idx+1]):
        #             tokens[idx+1] = type_convert(tokens[idx+1])
        #             res.append(tokens[idx]+tokens[idx+1])
        #             idx += 1
        #         else:
        #             tokens[idx] = type_convert(tokens[idx])
        #             res.append(tokens[idx])
        #     idx += 1
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
        return tokens

if __name__ == '__main__':
    # df = pd.read_csv('dataset.csv')
    # df['text'] = df['text'].apply(clean_text)
    # print(f"Before: {len(df)}")
    # df.drop_duplicates(['text'], inplace=True)
    # print(f"After: {len(df)}")
    print(Tokenize().tokenize(clean_text("a rt @abc wayyyy :) <abc <abc>>:((( $AAPL !!! www.abc.com 1125 115,000 +115,000 $11.25 -10.25- $aabc $35billion $15k 12asdf af12a 50-12 oo-af @abc 2018-12-31 -0.25 4.3 0.6 +1.2 $80-20 80/20 85/15/12 80.2- $50.50-")))