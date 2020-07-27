from torchtools import EarlyStopping
import argparse
import re
from tqdm import tqdm
from models.bert import BertClassifier
from models.BertbiLSTM import BertbiLSTM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import PreprocessedDataset, Dataset
from torchtext import data
from preprocess import Tokenize
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from utils import binary_accuracy, train, evaluate, epoch_time
from utils import plot_acc, plot_loss
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience of early stopping')
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    load_preprocessed_dataset = True
    max_length = 512
    bert_config = 'roberta-base'
    

    tokenizer = RobertaTokenizer.from_pretrained(bert_config)
    CLS_IDX = tokenizer.cls_token_id
    UNK_IDX = tokenizer.unk_token_id
    PAD_IDX = tokenizer.pad_token_id
    EOS_IDX = tokenizer.sep_token_id
    TEXT = data.Field(batch_first=True, use_vocab=False, preprocessing=tokenizer.convert_tokens_to_ids, 
            fix_length=max_length, init_token=CLS_IDX, eos_token=EOS_IDX, 
            pad_token=PAD_IDX, unk_token=UNK_IDX, lower=True)
    RAWFIELD = data.RawField()
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    fields = [
                ('date', RAWFIELD),
                ('text', TEXT),
                ('rise', LABEL),
            ]

    if load_preprocessed_dataset:
        train_dataset = PreprocessedDataset(path='final_dataset/preprocessed/dataset_train_pre.csv', fields=fields)
        test_dataset = PreprocessedDataset(path='final_dataset/preprocessed/dataset_test_pre.csv', fields=fields)
    else:
        train_dataset = Dataset(path='final_dataset/dataset_train.csv', fields=fields)
        test_dataset = Dataset(path='final_dataset/dataset_test.csv', fields=fields)

    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    # max_grad_norm = 1.0
    # num_training_steps = 1000
    # num_warmup_steps = 100
    # warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    model = BertbiLSTM(bert_config, freeze=True)
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model = model.to(device)
    print("Number of parameters")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print()
    print("Hyperparameters:")
    print(f"Batch size: {args.batch_size}")
    print(f"lr: {args.lr}")
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # To reproduce BertAdam specific behavior set correct_bias=False
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    # BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=0.05, num_training_steps=num_training_steps)
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
                }, f'roberta-model-acc-{valid_acc}-{mcc}.pth')

        if early_stopping.early_stop:
            print(f"Epochs: {epoch} - Early Stopping...")
            break
    
    plot_acc(train_accs, valid_accs, fname=f"roberta-epochs-{epochs}-acc-{valid_acc}.png")
    plot_loss(train_losses, valid_losses, fname=f"roberta-epochs-{epochs}-loss-{valid_acc}.png")
