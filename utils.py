import torch
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import math

labels = []
preds = []

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def categorical_accuracy(pred, label):

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)

    batch_size = pred.size(0)
    acc = 0.
    corrects = []

    for i in range(batch_size):
        _, pred_topi = pred[i].topk(2)
        label_index = label[i].argmax()
        global labels
        global preds

        labels += [label_index.item()]

        if pred_topi[0] == 2:
            acc += (pred_topi[1] == label_index)
            corrects += [(pred_topi[1] == label_index).cpu().numpy().tolist()]

            preds += [pred_topi[1].item()]
        else:
            acc += (pred_topi[0] == label_index)
            preds += [pred_topi[0].item()]
            corrects += [(pred_topi[0] == label_index).cpu().numpy().tolist()]

    return acc / batch_size, corrects

def binary_accuracy(pred, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    corrects = []

    global labels
    global preds

    labels += y.clone().detach().view(-1).cpu().numpy().tolist()
    #round predictions to the closest integer
     
    rounded_preds = torch.round(torch.sigmoid(pred))
    preds += rounded_preds.detach().view(-1).cpu().numpy().tolist()
    correct = (rounded_preds == y).float() #convert into float for division
    corrects += (rounded_preds == y).cpu().numpy().tolist()
    acc = correct.sum() / len(correct)
    return acc, corrects

def cal_mcc(cf_matrix):
    tn, fp, fn, tp = cf_matrix.ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    mcc = numerator / denominator
    return mcc

def train(model, iterator, optimizer, 
        criterion, metric=categorical_accuracy):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator):
        optimizer.zero_grad()

        predictions = model(batch.text)

        if metric.__name__ == 'binary_accuracy':
            predictions = predictions.squeeze(1)

        loss = criterion(predictions, batch.rise)
        acc, _ = metric(predictions, batch.rise)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, 
            fields, metric=categorical_accuracy): 
    epoch_loss = 0
    epoch_acc = 0
    # TEXT = fields[1][1]
    # correct_text = None
    # wrong_text = None

    model.eval()

    labels.clear()
    preds.clear()

    with torch.no_grad():
        
        for batch in tqdm(iterator):
            predictions = model(batch.text)

            if metric.__name__ == 'binary_accuracy':
                predictions = predictions.squeeze(1)

            loss = criterion(predictions, batch.rise)
            acc, corrects = metric(predictions, batch.rise)
            assert len(corrects) == len(batch.rise)

            # datas = []

            # for text in batch.text:
            #     ori_data = ' '.join([TEXT.vocab.itos[token] for token in text])
            #     datas.append(ori_data)
            
            # for idx, correct in enumerate(corrects):
            #     if correct:
            #         correct_text = datas[idx]
            #     else:
            #         wrong_text = datas[idx]
                
            epoch_loss += loss.item()
            epoch_acc += acc
    cm = confusion_matrix(labels, preds, labels=[1, 0])
    mcc = cal_mcc(cm)
    # print("correct predict text")
    # print(correct_text)
    # print()
    # print("wrong predict text")
    # print(wrong_text)
    # print()
    # print("confusion matrix")
    # print(cm)
    print(f"MCC: {mcc}")
    print()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), mcc

def plot_loss(train_loss, val_loss, fname='loss.png'):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Val. loss')
    plt.legend()
    plt.savefig(fname)

def plot_acc(train_acc, val_acc, fname='acc.png'):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Acc.')
    plt.plot(train_acc, label='Train Acc.')
    plt.plot(val_acc, label='Val. Acc.')
    plt.legend()
    plt.savefig(fname)