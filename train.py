import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig
from dataset import ReviewDataset
from model import ReviewsModel

EPOCHS = 10
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 3e-5

DATA_PATH = 'data'
SAVE_PATH = 'weights'
MODELNAME = 'bert-reviews-test'
MODELNAME_OR_PATH = 'bert-base-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tokenizer = AutoTokenizer.from_pretrained(MODELNAME_OR_PATH)
config = AutoConfig.from_pretrained(MODELNAME_OR_PATH)
base_model = AutoModel.from_pretrained(MODELNAME_OR_PATH, config=config)

trainset = ReviewDataset(tokenizer, DATA_PATH, MAX_LENGTH, do_train=True)
testset = ReviewDataset(tokenizer, DATA_PATH, MAX_LENGTH, do_train=False)

model = ReviewsModel(base_model)
model.to(device)

total = sum(p.numel() for p in model.parameters())
print(f'total parameters: {total:,}')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

class Tracker:
    def __init__(self, model_name, save_path):
        self.model_name = model_name
        self.save_path = save_path
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model = None

        self.last_epoch = 0
        self.last_best_epoch = 0
        
        self.log_path = os.path.join(self.save_path, self.model_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.log_path = os.path.join(self.log_path, 'log.txt')
        if os.path.exists(self.log_path):
            self.get_last_best()

        self.log = open(self.log_path, 'a+')

        print("last epoch:", self.last_epoch)
        if self.last_best_epoch:
            print("last best epoch:", self.last_best_epoch)

    def update(self, epoch, loss, acc, model):
        if loss < self.best_loss:
            self.write_log(epoch, loss, acc, is_best=True)
            self.best_loss = loss
            self.best_acc = acc
            self.best_epoch = epoch
            print(f'    - best loss: {self.best_loss:.4f} acc: {self.best_acc:.4f} epoch: {self.best_epoch}')
            self.save(model)
        else:
            self.write_log(epoch, loss, acc)

    def save(self, model):
        model.save_pretrained(os.path.join(self.save_path, self.model_name, 'best'))

    def load(self):
        self.best_model.save_pretrained(os.path.join(self.save_path, self.model_name))
    
    def write_log(self, epoch, loss, acc, is_best=False):
        if not is_best:
            self.log.write(f'{epoch:4d} {loss:.4f} {acc:.4f}\n')
        else:
            self.log.write(f'{epoch:4d} {loss:.4f} {acc:.4f} *\n')
        self.log.flush()

    def get_last_best(self):
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
            lines.reverse()
            if not len(lines): return
            last_line = lines[0].strip().split()
            if not len(last_line): return
            self.last_epoch = int(last_line[0])
            for line in lines:
                if '*' in line:
                    line = line.strip().split()
                    if not len(line): return
                    self.last_best_epoch = int(line[0])
                    self.best_loss = float(line[1])
                    self.best_acc = float(line[2])
                    return

    def get_epoch(self):
        return self.last_epoch + 1

    def close(self):
        self.log.close()

def train(epoch, model, trainloader, optimizer, criterion, verbose=1):
    model.train()
    train_loss = []
    train_acc = []
    show_iteration = len(trainloader) // 20
    bar = tqdm(trainloader) if verbose==1 else trainloader
    for batch_idx, batch in enumerate(bar):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels, ratings = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).sum().item() / len(labels)
        train_acc.append(acc)
        if verbose==1:
            bar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} Acc: {:.2f} %'.format(
                epoch, batch_idx, len(trainloader),
                100. * batch_idx / len(trainloader), 
                torch.mean(torch.tensor(train_loss[-10:]).float()), torch.mean(torch.tensor(train_acc[-10:]).float())*100))
        elif verbose==2:
            if batch_idx % show_iteration == 0:
                s = 'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} Acc: {:.2f} %'.format(
                    epoch, batch_idx, len(trainloader),
                    100. * batch_idx / len(trainloader), 
                    torch.mean(torch.tensor(train_loss[-10:]).float()), torch.mean(torch.tensor(train_acc[-10:]).float())*100)
                print(s)

def eval(epoch, model, testloader, criterion, verbose=1):
    model.eval()
    test_loss = []
    test_acc  = []
    show_iteration = len(testloader) // 20
    bar = tqdm(testloader) if verbose==1 else testloader
    for batch_idx, batch in enumerate(bar):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels, ratings = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, ratings)
        test_loss.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).sum().item() / len(labels)
        test_acc.append(acc)
        if verbose==1:
            bar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} Acc: {:.2f} %'.format(
                epoch, batch_idx, len(testloader),
                100. * batch_idx / len(testloader), 
                torch.mean(torch.tensor(test_loss).float()), torch.mean(torch.tensor(test_acc).float())*100))
        elif verbose==2:
            if batch_idx % show_iteration == 0:
                s = 'Test Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} Acc: {:.2f} %'.format(
                    epoch, batch_idx, len(testloader),
                    100. * batch_idx / len(testloader), 
                    torch.mean(torch.tensor(test_loss).float()), torch.mean(torch.tensor(test_acc).float())*100)
                print(s)

    return torch.mean(torch.tensor(test_loss)), torch.mean(torch.tensor(test_acc))

model.from_pretrained('weights/bert-reviews-kaggle')

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

tracker = Tracker(MODELNAME, SAVE_PATH)

import numpy as np

for epoch in range(tracker.get_epoch(), EPOCHS+1):
    # train(epoch, model, trainloader, optimizer, criterion)
    # test_loss, test_acc = eval(epoch, model, testloader, criterion)
    test_loss = np.random.rand()
    test_acc = np.random.rand()
    print(f'Epoch {epoch}: Test loss: {test_loss} Test acc: {test_acc}')

    tracker.update(epoch, test_loss, test_acc, model)
    print('=' * 50, '\n')

    model.save_pretrained(os.path.join(SAVE_PATH, MODELNAME))

tracker.close()       

# eval after training
# model.from_pretrained(os.path.join(SAVE_PATH, MODELNAME))
# test_loss, test_acc = eval(EPOCHS, model, testloader, criterion)