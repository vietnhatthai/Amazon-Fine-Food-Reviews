import os
import torch
from tqdm import tqdm

class Tracker:
    def __init__(self, model_name, save_path, model=None):
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
            self._get_last_best()

        self.log = open(self.log_path, 'a+')

        print("last epoch:", self.last_epoch)
        if self.last_best_epoch:
            print("last best epoch:", self.last_best_epoch)
        if self.last_epoch and model is not None:
            self._load_last_model(model)

    def update(self, epoch, loss, acc, model):
        if loss < self.best_loss:
            self._write_log(epoch, loss, acc, is_best=True)
            self.best_loss = loss
            self.best_acc = acc
            self.best_epoch = epoch
            print(f'    - best loss: {self.best_loss:.4f} acc: {self.best_acc:.4f} epoch: {self.best_epoch}')
            self._save(model)
        else:
            self._write_log(epoch, loss, acc)

    def _save(self, model):
        model.save_pretrained(os.path.join(self.save_path, self.model_name, 'best'))

    def _load_last_model(self, model):
        if (os.path.exists(os.path.join(self.save_path, self.model_name, 'config.json')) 
        and os.path.exists(os.path.join(self.save_path, self.model_name, 'pytorch_model.dat'))):
            model.from_pretrained(os.path.join(self.save_path, self.model_name))
    
    def _write_log(self, epoch, loss, acc, is_best=False):
        if not is_best:
            self.log.write(f'{epoch:4d} {loss:.4f} {acc:.4f}\n')
        else:
            self.log.write(f'{epoch:4d} {loss:.4f} {acc:.4f} *\n')
        self.log.flush()

    def _get_last_best(self):
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
        batch = tuple(t.to(model.get_device()) for t in batch)
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
        batch = tuple(t.to(model.get_device()) for t in batch)
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

def test_eval(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = tuple(t.to(model.get_device()) for t in batch)
            input_ids, attention_mask, labels, ratings = batch
            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.detach().cpu().numpy())
    return preds