import torch
import asyncio
from torch import nn

class WordsGRUHybridSent(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16 
        self.in_features = in_features
        self.out_features = out_features
 
        self.gru = nn.GRU(in_features, self.hidden_size, batch_first=True, bidirectional=True, num_layers = 5, dropout = 0.1)
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True, bidirectional=True, num_layers = 5, dropout = 0.1)
        self.lstm = nn.LSTM (in_features, self.hidden_size, batch_first=True, bidirectional=True, num_layers = 5, dropout = 0.1)

        self.mid = nn.Linear(self.hidden_size * 6, self.hidden_size)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.hidden_size, out_features)
 
    def forward(self, x):
        x1, h = self.gru(x)
        x2, h2 = self.rnn(x)
        x3, (h_n, c_n) = self.lstm(x)
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        hh2 = torch.cat((h2[-2, :, :], h2[-1, :, :]), dim=1)
        hh3 = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        hh = torch.cat((hh,hh2,hh3),dim = 1)
        y = self.mid(hh)
        y = self.drop(y)
        y = torch.relu(y)
        y = self.out(y)
        return y
    


class WordsGRUHybridPerc(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16 
        self.in_features = in_features
        self.out_features = out_features
 
        self.gru = nn.GRU(in_features, self.hidden_size, batch_first=True, bidirectional=True, num_layers = 5, dropout = 0.1)
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True, bidirectional=True, num_layers = 5, dropout = 0.1)
        self.lstm = nn.LSTM (in_features, self.hidden_size, batch_first=True, bidirectional=True, num_layers = 5, dropout = 0.1)

        self.mid = nn.Linear(self.hidden_size * 6 + 1, self.hidden_size )
        self.norm = nn.BatchNorm1d(self.hidden_size )
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.hidden_size , out_features)
 
    def forward(self, x, x_sent):
        x1, h = self.gru(x)
        x2, h2 = self.rnn(x)
        x3, (h_n, c_n) = self.lstm(x)
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        hh2 = torch.cat((h2[-2, :, :], h2[-1, :, :]), dim=1)
        hh3 = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        hh = torch.cat((hh,hh2,hh3),dim = 1)
        hh = torch.cat((hh, x_sent), dim=1)
        y = self.mid(hh)
        y = torch.relu(y)
        y = self.norm(y)
        y = self.drop(y)
        y = self.out(y)
        return (torch.sigmoid(y) * 2 - 1) * 50
    

def GetNeuroSent(mod_sent, navec, device, phrase):
    phrase_lst = phrase.lower().split()
    phrase_lst = [torch.tensor(navec[w]) for w in phrase_lst if w in navec]
    if (len(phrase_lst) == 0): # защита от исключений, вызванных бессмысленным или нулевым вводом пользователя
        return 0.5
    _data_batch = torch.stack(phrase_lst).to(device)
    predict = mod_sent(_data_batch.unsqueeze(0)).squeeze(0)
    p = torch.nn.functional.sigmoid(predict).item()
    return p

def GetNeuroPerc(mod_perc, sent, navec, device, phrase):
    phrase_lst = phrase.lower().split()
    phrase_lst = [torch.tensor(navec[w]) for w in phrase_lst if w in navec]
    if (len(phrase_lst) == 0):
        return 0.5
    _data_batch = torch.stack(phrase_lst).to(device)

    sentiment_value = sent
    sentiment_tensor = torch.tensor([[sentiment_value]], dtype=torch.float32).to(device)  # 1x1

    predict = mod_perc(_data_batch.unsqueeze(0), sentiment_tensor).squeeze(0)
    
    return predict.item()



