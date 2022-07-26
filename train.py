import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig
from dataset import ReviewDataset
from model import ReviewsModel
from utils import Tracker, train, eval, test_eval

EPOCHS = 10
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 3e-6

LANG = 'vi'
DATA_PATH = 'data'
SAVE_PATH = 'weights'
MODELNAME_OR_PATH = 'vinai/phobert-base'
MODELNAME = MODELNAME_OR_PATH.split('/')[-1] + '-reviews' + '-' + LANG

print("Model name:", MODELNAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tokenizer = AutoTokenizer.from_pretrained(MODELNAME_OR_PATH)
config = AutoConfig.from_pretrained(MODELNAME_OR_PATH)
base_model = AutoModel.from_pretrained(MODELNAME_OR_PATH, config=config)

trainset = ReviewDataset(tokenizer, DATA_PATH, MAX_LENGTH, do_train=True, lang=LANG)
testset = ReviewDataset(tokenizer, DATA_PATH, MAX_LENGTH, do_train=False, lang=LANG)

model = ReviewsModel(base_model, tokenizer)
model.to(device)

total = sum(p.numel() for p in model.parameters())
print(f'total parameters: {total:,}')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

tracker = Tracker(MODELNAME, SAVE_PATH, model)

for epoch in range(tracker.get_epoch(), EPOCHS+1):
    train(epoch, model, trainloader, optimizer, criterion)
    test_loss, test_acc = eval(epoch, model, testloader, criterion)

    print(f'Epoch {epoch}: Test loss: {test_loss} Test acc: {test_acc}')

    tracker.update(epoch, test_loss, test_acc, model)
    print('=' * 50, '\n')

    model.save_pretrained(os.path.join(SAVE_PATH, MODELNAME))
    tokenizer.save_pretrained(os.path.join(SAVE_PATH, MODELNAME))

tracker.close()       



model = ReviewsModel()

if os.path.exists(os.path.join(SAVE_PATH, MODELNAME, 'best')):
    model.from_pretrained(os.path.join(SAVE_PATH, MODELNAME, 'best'))
else:
    model.from_pretrained(os.path.join(SAVE_PATH, MODELNAME))

model.to(device)
total = sum(p.numel() for p in model.parameters())
print(f'total parameters: {total:,}')

trainloader = DataLoader(trainset, batch_size=512, shuffle=False)
train_preds = test_eval(model, trainloader)

# save train predictions
with open('train_preds.txt', 'w') as f:
    for pred in train_preds:
        f.write(str(pred) + '\n')

testloader = DataLoader(testset, batch_size=512, shuffle=False)
test_preds = test_eval(model, testloader)

# save test predictions
with open('test_preds.txt', 'w') as f:
    for pred in test_preds:
        f.write(str(pred) + '\n')