from torchtext import data
from dataset import Dataset, make_label, PreprocessedDataset
from models.TEGRU import TEGRU
import argparse
import torch
import  torch.nn as nn
from utils import evaluate, binary_accuracy, cal_mcc
import pandas as pd
import ast
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predicts(model, path, text, output_path='.'):
    model.eval()
    test_df = pd.read_csv(path)
    
    with torch.no_grad():
        predict_data = {'date': [], 'text': [], 'rise': []}
        
        for _, row in test_df.iterrows():
            date = row["date"]
            tweet = ast.literal_eval(row['text'])
            
            indexed = [text.vocab.stoi[token] for token in tweet]
            input = torch.LongTensor(indexed).to(device)
            input = input.unsqueeze(0)
            
            prediction = torch.round(torch.sigmoid(model(input))).item()
            predict_data['date'].append(date)
            predict_data['text'].append(str(tweet))
            predict_data['rise'].append(prediction)
        
        df = pd.DataFrame(data=predict_data)
        df.to_csv(f"{output_path}/{path.split('/')[-1].split('.')[0] + '_pred.csv'}", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_path', type=str, help='from given path load pretrained model')
    parser.add_argument('--test_file', type=str, help='testfile path')
    
    args = parser.parse_args()

    pretrained_model_args = torch.load(args.model_path)
    TEXT = pretrained_model_args['text']
    RAWFIELD = data.RawField()
    embed_dim = pretrained_model_args['dim']
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    load_model = TEGRU(pretrained_vec=TEXT.vocab.vectors, ntoken=len(TEXT.vocab), d_model=embed_dim, 
                    nhid=512, pad_token_id=PAD_IDX, hidden_dim=256, 
                    te_nlayers=6)
    load_model = load_model.to(device)
    load_model.load_state_dict(pretrained_model_args['net'])

    if args.test_file:
        predicts(load_model, args.test_file, TEXT)
    else:
        LABEL = data.LabelField(dtype=torch.float)
        fields = [
                    ('date', RAWFIELD),
                    ('text', TEXT),
                    ('rise', LABEL),
                ]
        test_dataset = PreprocessedDataset(path='final_dataset/preprocessed/dataset_test_pre.csv', fields=fields)
        LABEL.build_vocab(test_dataset)
        criterion = nn.BCEWithLogitsLoss()
        criterion = criterion.to(device)
        load_model.load_state_dict(pretrained_model_args['net'])
        valid_iterator = data.Iterator(test_dataset, args.batch_size, train=False, sort=False, shuffle=True, device=device)
        valid_loss, valid_acc, _ = evaluate(load_model, valid_iterator, criterion, fields=fields, metric=binary_accuracy)

        print(f'\t Val. Loss: {valid_loss:.6f} |  Val. Accuracy: {valid_acc}')
