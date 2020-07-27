from torchtools import EarlyStopping
import argparse
from models.TEbiLSTM import TEbiLSTM
from models.TEGRU import TEGRU
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import Dataset, PreprocessedDataset
from torchtext import data
from preprocess import Tokenize
import numpy as np
from utils import binary_accuracy, epoch_time
from utils import train, evaluate, plot_acc, plot_loss, cal_mcc
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience of early stopping')
    parser.add_argument('--eval', type=str, help='load model and evaluate')
    parser.add_argument('--pretrained_w2v', action='store_true', help='load pretrained w2v')
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    load_preprocessed_dataset = True
    embed_dim = 200
    model = None
    max_length = 512

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    RAWFIELD = data.RawField()
    LABEL = data.LabelField(dtype=torch.float)
    fields = [
                ('date', RAWFIELD),
                ('text', TEXT),
                ('rise', LABEL),
            ]

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if load_preprocessed_dataset:
        dataset = PreprocessedDataset(path='final_dataset/preprocessed/dataset_train_pre.csv', fields=fields)
        train_dataset, valid_dataset = dataset.split(split_ratio=0.95, stratified=True, strata_field='rise')
        test_dataset = PreprocessedDataset(path='final_dataset/preprocessed/dataset_test_pre.csv', fields=fields)
    else:
        dataset = PreprocessedDataset(path='final_dataset/dataset_train_pre.csv', fields=fields)
        train_dataset, valid_dataset = dataset.split(split_ratio=0.95, stratified=True, strata_field='rise')
        test_dataset = PreprocessedDataset(path='final_dataset/dataset_test_pre.csv', fields=fields)

    if args.pretrained_w2v:
        TEXT.build_vocab(train_dataset,
                        vectors="glove.twitter.27B.200d",
                        unk_init=torch.Tensor.normal_, 
                        min_freq=2)
        LABEL.build_vocab(train_dataset)
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        model = TEGRU(pretrained_vec=TEXT.vocab.vectors, ntoken=len(TEXT.vocab), d_model=embed_dim, 
                    nhid=embed_dim*2, pad_token_id=PAD_IDX, te_nlayers=6)
        
    else:
        TEXT.build_vocab(train_dataset)
        LABEL.build_vocab(train_dataset)
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        model = TEGRU(ntoken=len(TEXT.vocab), d_model=embed_dim)

    model = model.to(device)
    print(model)
    print()
    print("Number of parameters")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print()
    print("Hyperparameters:")
    print(f"Batch size: {args.batch_size}")
    print(f"lr: {args.lr}")
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
    
    train_iterator = data.Iterator(train_dataset, batch_size, train=True, shuffle=True, device=device)
    valid_iterator = data.Iterator(test_dataset, batch_size, train=False, sort=False, shuffle=True, device=device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_acc = 0.
    valid_acc = 0.
    mcc = 0.
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, metric=binary_accuracy)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.6f} | Train Accuracy: {train_acc}')
        
        valid_loss, valid_acc, mcc = evaluate(model, valid_iterator, criterion, fields=fields, metric=binary_accuracy)
        print(f'\t Val. Loss: {valid_loss:.6f} |  Val. Accuracy: {valid_acc}')
        train_accs += [train_acc]
        train_losses += [train_loss]
        valid_accs += [valid_acc]
        valid_losses += [valid_loss]
        # scheduler.step()

        early_stopping(valid_loss, model)
        
        if valid_acc > 0.55 and mcc > 0.056:
            torch.save(
                {
                    'net': model.state_dict(),
                    'text': TEXT,
                    'dim': embed_dim,
                }, f'te-model-acc-{valid_acc}-{mcc}.pth')

        if early_stopping.early_stop:
            print(f"Epochs: {epoch} - Early Stopping...")
            break
    
    plot_acc(train_accs, valid_accs, fname=f"ternn-epochs-{epochs}-acc-{valid_acc}.png")
    plot_loss(train_losses, valid_losses, fname=f"ternn-epochs-{epochs}-loss-{valid_acc}.png")
    
        
    
    
