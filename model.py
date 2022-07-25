import json
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ReviewsModel(nn.Module):
    def __init__(self, base_model=None, num_classes=5):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        self.h_RNN_layers = 3       # RNN hidden layers
        self.h_RNN = 256            # RNN hidden nodes

        if base_model is not None:
            self._get_model()

    def _get_model(self):
        self.last_hidden_layer_size = self.base_model.config.hidden_size
        self.lstm = nn.LSTM(
                            input_size=self.last_hidden_layer_size,
                            hidden_size=self.h_RNN,        
                            num_layers=self.h_RNN_layers,       
                            batch_first=True,           # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
                            )

        self.fc = nn.Linear(self.h_RNN, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):

        self.lstm.flatten_parameters()

        if self.base_model is None:
            raise Exception('Model is not initialized')

        x = self.base_model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            return_dict=True)
        
        RNN_out, (h_n, h_c) = self.lstm(x.last_hidden_state, None)  
        
        x = self.fc(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = self.softmax(x)
        return x

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'pytorch_model.dat')
        torch.save(self.state_dict(), model_path)
        config = self.base_model.config
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            f.write(config.to_json_string())
    
    def from_pretrained(self, path):
        model_path = os.path.join(path, 'pytorch_model.dat')
        device = self.base_model.device if self.base_model is not None else 'cpu'
        self.base_model = AutoModel.from_config(AutoConfig.from_pretrained(path))
        self._get_model()

        self.load_state_dict(torch.load(model_path))
        self.to(device)
        self.eval()


def test():
    from transformers import AutoTokenizer
    
    MODELNAME_OR_PATH = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODELNAME_OR_PATH)
    
    config = AutoConfig.from_pretrained(MODELNAME_OR_PATH)
    base_model = AutoModel.from_pretrained(MODELNAME_OR_PATH, config=config)

    model = ReviewsModel(base_model)

    # model.save_pretrained('weights/bert_reviews')
    # model.from_pretrained('weights/bert_reviews')
    
    model.eval()

    token = tokenizer.encode_plus('My cat is very cute',
                                add_special_tokens=True,    
                                max_length=32,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')

    ids = token['input_ids']
    mask = token['attention_mask']
    with torch.no_grad():
        output = model(ids, mask)
    print(output)

if __name__=='__main__':
    test()