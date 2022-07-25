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
MODELNAME = 'bert-reviews'
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

def train(epoch, model, trainloader, optimizer, criterion, verbose=True):
    model.train()
    train_loss = []
    train_acc = []
    bar = tqdm(trainloader) if verbose else trainloader
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
        if verbose:
            bar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} Acc: {:.2f} %'.format(
                epoch, batch_idx, len(trainloader),
                100. * batch_idx / len(trainloader), 
                torch.mean(torch.tensor(train_loss[-10:]).float()), torch.mean(torch.tensor(train_acc[-10:]).float())*100))

def eval(epoch, model, testloader, criterion, verbose=True):
    model.eval()
    test_loss = []
    test_acc  = []
    bar = tqdm(testloader) if verbose else testloader
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
        if verbose:
            bar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} Acc: {:.2f} %'.format(
                epoch, batch_idx, len(testloader),
                100. * batch_idx / len(testloader), 
                torch.mean(torch.tensor(test_loss).float()), torch.mean(torch.tensor(test_acc).float())*100))
    return torch.mean(torch.tensor(test_loss)), torch.mean(torch.tensor(test_acc))

model.from_pretrained('weights/bert-reviews-kaggle')

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

# for epoch in range(EPOCHS):
#     train(epoch, model, trainloader, optimizer, criterion)
#     test_loss, test_acc = eval(epoch, model, testloader, criterion)

#     print(f'Test loss: {test_loss} Test acc: {test_acc}')
#     print('-' * 10)

#     model.save_pretrained(os.path.join(SAVE_PATH, MODELNAME))

# eval after training
# model.from_pretrained(os.path.join(SAVE_PATH, MODELNAME))
test_loss, test_acc = eval(EPOCHS, model, testloader, criterion)