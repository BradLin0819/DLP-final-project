import pandas as pd
import torch
import torchtext
from torchtext import data
from preprocess import Tokenize
import numpy as np
import datetime
import ast

def make_label(label):
    labels = np.zeros(3)
    labels[label] = 1
    return labels

class Dataset(data.Dataset):
    def __init__(self, path, fields, tokenize=Tokenize().tokenize,
                tweet_tokens_len=20, tweet_per_day=45, eos_token='<eos>',
                pad_token='<pad>', **kwargs):
        df = pd.read_csv(path)
        df.drop_duplicates(["text"], inplace=True)
        total_days = df["date"].apply(lambda x: x.split()[0]).unique()
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M:%S")
        total_companies = df["company"].unique()
        examples = []

        for company in total_companies:
            for day in total_days:
                date = day
                day = list(map(int, day.split('-')))
                filter_mask = (df["company"] == company) & (df["date"].dt.date == datetime.date(*day))
                filter_df = df[filter_mask].sort_values(by=["date"])

                if len(filter_df) > tweet_per_day:
                    filter_df = filter_df.iloc[-tweet_per_day:]

                seq_len = tweet_tokens_len * tweet_per_day
                final_text = []
                label = None

                for idx, row in filter_df.iterrows():
                    label = row['rise']
                    tweet_tokens = tokenize(row['text'])
                    real_tweet_tokens_len = len(tweet_tokens)

                    if real_tweet_tokens_len >= tweet_tokens_len:
                        tweet_tokens = tweet_tokens[:tweet_tokens_len]
                    else:
                        pad_tweet_tokens = [pad_token] * (tweet_tokens_len - len(tweet_tokens))
                        tweet_tokens = (pad_tweet_tokens + tweet_tokens)

                    assert len(tweet_tokens) == tweet_tokens_len
                    final_text += tweet_tokens

                if label is not None:
                    if len(final_text) < seq_len:
                        padding = [pad_token] * (seq_len - len(final_text))
                        final_text += padding

                    assert len(final_text) == seq_len
                    examples.append(data.Example.fromlist([date, final_text, int(label)], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

class Testset(data.Dataset):
    def __init__(self, path, fields, tokenize=Tokenize().tokenize,
                tweet_tokens_len=20, tweet_per_day=45, eos_token='<eos>',
                pad_token='<pad>', **kwargs):
        df = pd.read_csv(path)
        df.drop_duplicates(["text"], inplace=True)
        total_days = df["date"].apply(lambda x: x.split()[0]).unique()
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M:%S")
        examples = []

        for day in total_days:
            date = day
            day = list(map(int, day.split('-')))
            filter_mask = (df["date"].dt.date == datetime.date(*day))
            filter_df = df[filter_mask].sort_values(by=["date"])

            if len(filter_df) > tweet_per_day:
                filter_df = filter_df.iloc[-tweet_per_day:]

            seq_len = tweet_tokens_len * tweet_per_day
            final_text = []


            for idx, row in filter_df.iterrows():
                tweet_tokens = tokenize(row['text'])
                real_tweet_tokens_len = len(tweet_tokens)

                if real_tweet_tokens_len >= tweet_tokens_len:
                    tweet_tokens = tweet_tokens[:tweet_tokens_len]
                else:
                    pad_tweet_tokens = [pad_token] * (tweet_tokens_len - len(tweet_tokens))
                    tweet_tokens = (pad_tweet_tokens + tweet_tokens)

                assert len(tweet_tokens) == tweet_tokens_len
                final_text += tweet_tokens

            if len(final_text) < seq_len:
                padding = [pad_token] * (seq_len - len(final_text))
                final_text += padding

            assert len(final_text) == seq_len
            examples.append(data.Example.fromlist([date, final_text], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

class ConcatDataset(data.Dataset):
    def __init__(self, path, fields, tweet_per_day=50, **kwargs):
        df = pd.read_csv(path)
        df.drop_duplicates(["text"], inplace=True)
        total_days = df["date"].apply(lambda x: x.split()[0]).unique()
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M:%S")
        total_companies = df["company"].unique()
        examples = []

        for company in total_companies:
            for day in total_days:
                date = day
                day = list(map(int, day.split('-')))
                filter_mask = (df["company"] == company) & (df["date"].dt.date == datetime.date(*day))
                filter_df = df[filter_mask].sort_values(by=["date"])
                filter_df = filter_df.iloc[-tweet_per_day:]
                final_text = ""
                label = None

                for idx, row in filter_df.iterrows():
                    label = row['rise']
                    final_text += (row['text'] + " ")

                if label is not None:
                    examples.append(data.Example.fromlist([date, final_text, int(label)], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

class PreprocessedDataset(data.Dataset):
    def __init__(self, path, fields, **kwargs):
        df = pd.read_csv(path)
        examples = []

        for idx, row in df.iterrows():
            date = row['date']
            text = ast.literal_eval(row['text'])
            label = row['rise']
            examples.append(data.Example.fromlist([date, text, int(label)], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

if __name__ == '__main__':
    TEXT = data.Field(sequential=True, batch_first=True)
    RAWFIELD = data.RawField()
    LABEL = data.Field(sequential=False, preprocessing=make_label, use_vocab=False)
    fields = [
                ('date', RAWFIELD),
                ('text', TEXT),
                ('rise', LABEL),
            ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PreprocessedDataset(path='./test_pre.csv', fields=fields)
    MAX_VOCAB_SIZE = 50000
    TEXT.build_vocab(dataset, 
                    max_size=MAX_VOCAB_SIZE, 
                    vectors="glove.6B.100d",
                    unk_init=torch.Tensor.normal_)
    print(vars(TEXT.vocab))
    print(TEXT.vocab.stoi['hello'])
