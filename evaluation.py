import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import ReviewDataset
from model import ReviewsModel

BATCH_SIZE = 128
MAX_LENGTH = 128

DATA_PATH = 'data'
SAVE_PATH = 'weights'
MODELNAME = 'bert-reviews-kaggle'
MODELNAME_OR_PATH = 'bert-base-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tokenizer = AutoTokenizer.from_pretrained(MODELNAME_OR_PATH)

trainset = ReviewDataset(tokenizer, DATA_PATH, MAX_LENGTH, do_train=True)
testset = ReviewDataset(tokenizer, DATA_PATH, MAX_LENGTH, do_train=False)

model = ReviewsModel()

if os.path.exists(os.path.join(SAVE_PATH, MODELNAME, 'best')):
    model.from_pretrained(os.path.join(SAVE_PATH, MODELNAME, 'best'))
else:
    model.from_pretrained(os.path.join(SAVE_PATH, MODELNAME))

model.to(device)
total = sum(p.numel() for p in model.parameters())
print(f'total parameters: {total:,}')

def eval(dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, ratings = batch
            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.detach().cpu().numpy())
    return preds

trainloader = DataLoader(trainset, batch_size=512, shuffle=False)
train_preds = eval(trainloader)

# save train predictions
with open('train_preds.txt', 'w') as f:
    for pred in train_preds:
        f.write(str(pred) + '\n')

testloader = DataLoader(testset, batch_size=512, shuffle=False)
test_preds = eval(testloader)

# save test predictions
with open('test_preds.txt', 'w') as f:
    for pred in test_preds:
        f.write(str(pred) + '\n')