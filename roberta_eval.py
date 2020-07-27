from torchtext import data
from dataset import Dataset, PreprocessedDataset
from models.BertbiLSTM import BertbiLSTM
import argparse
import torch
import  torch.nn as nn
from utils import evaluate, binary_accuracy, cal_mcc
import pandas as pd
import ast
from tqdm import tqdm
from transformers import RobertaTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = 512 
bert_config = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(bert_config)

def predicts(model, path, output_path='roberta_predictions'):
    model.eval()
    test_df = pd.read_csv(path)
    
    with torch.no_grad():
        predict_data = {'date': [], 'text': [], 'rise': []}
        
        for _, row in test_df.iterrows():
            date = row["date"]
            tweet = ast.literal_eval(row['text'])
            tweet = tweet[:max_length-2]
            indexed = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tweet) + [tokenizer.sep_token_id]
            input = torch.LongTensor(indexed).to(device)
            input = input.unsqueeze(0)
            
            prediction = torch.round(torch.sigmoid(model(input))).item()
            predict_data['date'].append(date)
            predict_data['text'].append(str(tweet))
            predict_data['rise'].append(prediction)
        
        df = pd.DataFrame(data=predict_data)
        df.to_csv(f"{output_path}/{path.split('/')[-1]}", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_path', type=str, help='from given path load pretrained model')
    parser.add_argument('--test_file', type=str, help='testfile path')
    
    args = parser.parse_args()

    pretrained_model_args = torch.load(args.model_path)
    CLS_IDX = tokenizer.cls_token_id
    UNK_IDX = tokenizer.unk_token_id
    PAD_IDX = tokenizer.pad_token_id
    EOS_IDX = tokenizer.sep_token_id
    TEXT = data.Field(batch_first=True, use_vocab=False, preprocessing=tokenizer.convert_tokens_to_ids, 
            fix_length=max_length, init_token=CLS_IDX, eos_token=EOS_IDX, 
            pad_token=PAD_IDX, unk_token=UNK_IDX, lower=True)
    RAWFIELD = data.RawField()

    load_model = BertbiLSTM(bert_config, freeze=True)
    load_model = load_model.to(device)
    load_model.load_state_dict(pretrained_model_args['net'])

    if args.test_file:
        predicts(load_model, args.test_file, TEXT)
    else:
        LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        fields = [
                    ('date', RAWFIELD),
                    ('text', TEXT),
                    ('rise', LABEL),
                ]
        test_dataset = PreprocessedDataset(path='final_dataset/preprocessed/dataset_test_pre.csv', fields=fields)
        criterion = nn.BCEWithLogitsLoss()
        criterion = criterion.to(device)
        load_model.load_state_dict(pretrained_model_args['net'])
        valid_iterator = data.Iterator(test_dataset, args.batch_size, train=False, sort=False, shuffle=True, device=device)
        valid_loss, valid_acc, _ = evaluate(load_model, valid_iterator, criterion, fields=fields, metric=binary_accuracy)

        print(f'\t Val. Loss: {valid_loss:.6f} |  Val. Accuracy: {valid_acc}')
