import os
import pandas as pd

import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, tokenizer, data_root, max_len, do_train=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.max_len = max_len
        self.split = 'train' if do_train else 'test'
        self.data = self._read_csv()

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        score = self.data.iloc[idx]['Score']
        rating = self.data.iloc[idx]['Rating']

        token = self._get_token(text)
        score = torch.FloatTensor([score])
        rating = torch.FloatTensor([rating])

        return token['input_ids'][0], token['attention_mask'][0], score, rating

    def __len__(self):
        return len(self.data)

    def _get_token(self, text):
        return self.tokenizer.encode_plus(text, 
                                    add_special_tokens=True, 
                                    max_length=self.max_len,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')

    def _read_csv(self):
        return pd.read_csv(os.path.join(self.data_root, self.split + '.csv'))
    
    def _bufcount(self):
        '''
        counting the number of lines in the file
        '''
        f = open(os.path.join(self.data_root, self.split + '.csv'), 'r')                 
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.read # loop optimization

        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)

        return lines

def test():
    from transformers import AutoTokenizer
    
    MODELNAME_OR_PATH = 'bert-base-uncased'
    MAX_LENGTH = 128

    tokenizer = AutoTokenizer.from_pretrained(MODELNAME_OR_PATH)
    dataset = ReviewDataset(tokenizer, 'data', MAX_LENGTH)

    print(dataset.data.head())

    data = dataset[0]
    print(data)

if __name__ == '__main__':
    test()