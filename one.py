#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
subprocess.call(['pip','install','wandb'])


# In[ ]:


import torch
from io import open
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb
import unicodedata
import string
import re
import random
import time
import torch.nn as nn
#wandb.login()
wandb.login()
#!wandb login --relogin
#!wandb login --relogin
import argparse
import wandb
import torch.nn as nn
from train import Encoder, Decoder, Train


# In[ ]:


sos = 0
eos = 1
# tokenization 
class Lang:
    def __init__(self, name):
        self.name = name
        self.letter_to_index = {}
        self.letter_to_count = {}
        self.index_to_letter = {0: "SOS", 1: "EOS"}
        self.n_letters = 2  

    def add_letter(self, letter): 
        if letter not in self.letter_to_index:
            self.letter_to_index[letter] = self.n_letters
            self.letter_to_count[letter] = 1
            self.index_to_letter[self.n_letters] = letter
            self.n_letters = self.n_letters+1
        else:
            self.letter_to_count[letter] = self.letter_to_count[letter]+1

    def add_word(self, letter): 
        for letter in letter:
            self.add_letter(letter)

#    def decode(self, target):
#        return ' 'join([self.index_to_letter[i.get] for i in target])
    def decode(self, target):
        words = []
        for i in target:
            words.append(self.index_to_letter[i.get])
        return ' '.join(words)



# In[ ]:


def readLang(lang1, lang2, reverse=False): # read the file and make a dictionary of words of both languages

    # Reading the uploaded file and split into lines
    train_lines = open('/home/user/Desktop/Dharani/hin_train (1).csv', encoding='utf-8').\
        read().strip().split('\n')
    val_lines = open('/home/user/Desktop/Dharani/hin_valid (1).csv', encoding='utf-8').\
        read().strip().split('\n')


    # Split every line into pairs and normalize
    train_pairs = []
    for l in train_lines:
        train_pairs.append(l.split(','))

    val_pairs = []
    for l in val_lines:
        val_pairs.append(l.split(','))
        
 


    inp_lang = Lang(lang1)
    out_lang = Lang(lang2)

    for pair in train_pairs:
        inp_lang.add_word(pair[0])
        out_lang.add_word(pair[1])
    
    for pair in val_pairs:
        inp_lang.add_word(pair[0])
        out_lang.add_word(pair[1])
        
   

    return train_pairs, val_pairs, inp_lang, out_lang


# In[ ]:


#def indexes_From_word(lang, word): # convert a word to a list of indexes
#    return [lang.letter_to_index[letter] for letter in word]

def indexes_From_word(lang, word):
    indexes_ = []
    for letter in word:
        indexes_.append(lang.letter_to_index[letter])
    return indexes_



def tensor_From_word(lang, word): # convert a word to a tensor
    indexes = indexes_From_word(lang, word)
    indexes.append(eos)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)




def tensor_From_Pair(pair, inp_lang, out_lang): # convert a pair of words to a pair of tensors
    inp_tensor = tensor_From_word(inp_lang, pair[0])
    Target_tensor = tensor_From_word(out_lang, pair[1])
    return (inp_tensor, Target_tensor)


# In[ ]:


class Encoder(nn.Module): #encoder processes input sequence(english)
    def __init__(self, rnn_type, inp_size, emb_size, hid_size, p, num_layers):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(inp_size, emb_size)
        self.rnn = nn.RNN(emb_size, hid_size, num_layers, dropout=p)
        self.gru = nn.GRU(emb_size, hid_size, num_layers, dropout=p)
        self.lstm = nn.LSTM(emb_size, hid_size, num_layers, dropout=p)
        self.rnn_type = rnn_type

    def forward(self, input, hidden):
        embedded_ = self.dropout(self.embedding(input)).view(1, 1, -1)  # embedding of word
        output = embedded_

        # giving output according to model type
        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(output, hidden)
        elif self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        elif self.rnn_type == 'LSTM':
            output, hidden = self.lstm(output, hidden)

        return output, hidden

    def initHidden(self):  # initializing hidden layer
        if self.rnn_type == 'LSTM':
            return (
                torch.zeros(self.num_layers, 1, self.hid_size, device=device),
                torch.zeros(self.num_layers, 1, self.hid_size, device=device),
            )
        return torch.zeros(self.num_layers, 1, self.hid_size, device=device)


# In[ ]:


class Decoder(nn.Module): #generates output sequence by utilizing context vector
    
    def __init__(self, rnn_type, out_size, embed_size, hid_size, p, num_layers):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.out_size = out_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(out_size, embed_size)
        self.rnn = nn.RNN(embed_size, hid_size, num_layers, dropout=p)
        self.gru = nn.GRU(embed_size, hid_size, num_layers, dropout=p)
        self.lstm = nn.LSTM(embed_size, hid_size, num_layers, dropout=p)
        self.out = nn.Linear(hid_size, out_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.rnn_type = rnn_type

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input)).view(1, 1, -1)     #embedding of words
        output = F.relu(output)                                         #applying activation to the input

        #output is given according to model type
        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(output, hidden)
        elif self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        elif self.rnn_type == 'LSTM':
            output, hidden = self.lstm(output, hidden)

        # softmax function to get probabilities
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):  # for initializing hidden layer
        if self.rnn_type == 'LSTM':
            return (
                torch.zeros(self.num_layers, 1, self.hid_size, device=device),
                torch.zeros(self.num_layers, 1, self.hid_size, device=device),
            )
        return torch.zeros(self.num_layers, 1, self.hid_size, device=device)


# In[ ]:


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[ ]:


class Train(): # training class
    def __init__(self, train_data, encoder, decoder, criterion, tf_ratio = 0.5):
        self.train_data = train_data
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.tf_ratio = tf_ratio
        self.train_pairs, self.val_pairs, self.inp_lang, self.out_lang = readLang('eng', 'hin')
        self.training_pairs = []
        for i in range(len(self.train_pairs)):
            pair = self.train_pairs[i]
            tensor_pair = tensor_From_Pair(pair, self.inp_lang, self.out_lang)
            self.training_pairs.append(tensor_pair)


    def train(self, inp_tensor, Target_tensor, encoder_opt, decoder_opt):
        encoder_hid = self.encoder.initHidden()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        encoder_outputs = torch.zeros(50, self.encoder.hid_size, device=device)

        loss = 0

        inp_len = inp_tensor.size(0)
        target_len = Target_tensor.size(0)

        for i in range(inp_len):                                                  # encoding a word
            encoder_output, encoder_hid = self.encoder(inp_tensor[i], encoder_hid)
            # print(encoder_output.shape)
            #print(encoder_hid.shape)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[sos]], device=device)
        decoder_hidden = encoder_hid                                              # encoder shares its hidden layer with decoder

        Teacher_Forcing = True if random.random() < self.tf_ratio else False

        if Teacher_Forcing: 
            for i in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss += self.criterion(decoder_output, Target_tensor[i])
                decoder_input = Target_tensor[i]                                  # teacher forcing

        else:
            for i in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)                               # topk predictions
                decoder_input = topi.squeeze().detach()                           # detach it from history as input
                loss += self.criterion(decoder_output, Target_tensor[i])
                if decoder_input.item() == eos:                                   # if EOS token is predicted, stop
                    break

        loss.backward() 
        encoder_opt.step()
        decoder_opt.step()

        return loss.item() / target_len


    def trainIters(self, optimizer, learning_rate, n_iters = 69, print_every = 69, epochs=-1): #optimizers 
        start = time.time()
        print_loss_total = 0

        if optimizer == 'SGD':
            encoder_opt = optim.SGD(self.encoder.parameters(), lr = learning_rate)
            decoder_opt = optim.SGD(self.decoder.parameters(), lr = learning_rate)
        elif optimizer == 'Adam':
            encoder_opt = optim.Adam(self.encoder.parameters(), lr = learning_rate)
            decoder_opt = optim.Adam(self.decoder.parameters(), lr = learning_rate)
        elif optimizer == 'RMSprop':
            encoder_opt = optim.RMSprop(self.encoder.parameters(), lr = learning_rate)
            decoder_opt = optim.RMSprop(self.decoder.parameters(), lr = learning_rate)
        elif optimizer == 'NAdam':
            encoder_opt = optim.NAdam(self.encoder.parameters(), lr = learning_rate)
            decoder_opt = optim.NAdam(self.decoder.parameters(), lr = learning_rate)

        if epochs != -1:
            n_iters = len(self.train_pairs)
        else:
            train_loss_total = 0
            for iter in tqdm(range(1, n_iters+1)):
                training_pair = self.training_pairs[iter - 1]
                inp_tensor = training_pair[0]
                Target_tensor = training_pair[1]
                loss = self.train(inp_tensor, Target_tensor, encoder_opt, decoder_opt)
                train_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
            train_acc = self.evaluateData(self.train_pairs)
            valid_acc = self.evaluateData(self.val_pairs)
            #test_acc = self.evaluateData(self.test_pairs)
            return train_acc, valid_acc

        Training_loss = []
        Val_accuracy = []
        Training_accuracy = []
        #Test_accuracy = []
        for j in range(epochs):
            train_loss_total = 0
            for iter in tqdm(range(1, n_iters+1)):
                training_pair = self.training_pairs[iter - 1]
                inp_tensor = training_pair[0]
                Target_tensor = training_pair[1]
                loss = self.train(inp_tensor, Target_tensor, encoder_opt, decoder_opt)
                train_loss_total += loss
                print_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
            train_acc = self.evaluateData(self.train_pairs)
            valid_acc = self.evaluateData(self.val_pairs)
            #test_acc = self.evaluateData(self.test_pairs)
            #print(val_accuracy)
            Training_loss.append(train_loss_total / n_iters)
            Val_accuracy.append(valid_acc)
            Training_accuracy.append(train_acc)
            #Test_accuracy.append(test_acc)
            wandb.log({'train_loss': train_loss_total / n_iters, 'train_acc': train_acc, 'valid_acc': valid_acc }) #loging to wandb for sweeps
        return Training_loss, Training_accuracy, Val_accuracy
                    

    def evaluate(self, word):
        with torch.no_grad():
            inp_tensor = tensor_From_word(self.inp_lang, word)
            inp_len = inp_tensor.size()[0]
            encoder_hid = self.encoder.initHidden()

            encoder_outputs = torch.zeros(50, self.encoder.hid_size, device=device)

            for i in range(inp_len): # encoding a word
                encoder_output, encoder_hid = self.encoder(inp_tensor[i], encoder_hid)
                # print(encoder_output.shape)
                encoder_outputs[i] += encoder_output[0, 0]

            decoder_input = torch.tensor([[sos]], device=device)
            decoder_hidden = encoder_hid # encoder shares its hidden layer with decoder

            decoded_word = ''

            for i in range(50):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1) # top k predictions
                if topi.item() == eos:
                    break
                else:
                    decoded_word += (self.out_lang.index_to_letter[topi.item()])
                decoder_input = topi.squeeze().detach() # detach from history as input

            return decoded_word
        
    def evaluateData(self, data): # defining function to determine accuracy while passing through arguments
        acc = 0
        total = len(data)  # Store the total number of data points
        for i in range(total):
            word, target = data[i]
            acc += (self.evaluate(word) == target)
        return acc / total

#train_pairs, val_pairs,inp_lang, out_lang = readLang('eng', 'hin')            


# In[ ]:


train_pairs, val_pairs,inp_lang, out_lang = readLang('eng', 'hin')


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(description='Run the script with hyperparameter configuration')
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD', 'Adam', 'RMSprop', 'NAdam'], help='Optimizer choice')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='Learning rate value')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--hid_layers', default=1, type=int, help='Number of hidden layers')
    parser.add_argument('--emb_size', default=256, type=int, help='Embedding size')
    parser.add_argument('--hid_size', default=256, type=int, help='Hidden size')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--type_t', default='GRU', type=str, choices=['RNN', 'LSTM', 'GRU'], help='Type of network')
    args = parser.parse_args()
    return args



def run(args):
    config_defaults = {
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'hid_layers': args.hid_layers,
        'emb_size': args.emb_size,
        'hid_size': args.hid_size,
        'dropout': args.dropout,
        'type_t': args.type_t
    }
    wandb.init(config=config_defaults)
    config = wandb.config
    encoder = Encoder(config.type_t, inp_lang.n_letters, config.emb_size, config.hid_size, config.dropout, config.hid_layers).to(device)
    decoder = Decoder(config.type_t, out_lang.n_letters, config.emb_size, config.hid_size, config.dropout, config.hid_layers).to(device)
    train = Train(train_pairs, encoder, decoder, nn.NLLLoss(), config.dropout)
    train.trainIters(config.optimizer, config.learning_rate, print_every=10000, epochs=config.epochs)
    wandb.finish()

    
if __name__ == '__main__':
    args = parse_args()
    run(args)


# In[ ]:


# # sweep configuration for wandb
# sweep_config = {
#     'method': 'random', 
#     'metric': {
#         'name': 'valid_acc',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'optimizer': {
#             'values': ['SGD', 'Adam', 'RMSprop', 'NAdam']
#         },
#         'learning_rate': {
#             'values': [1e-4, 5e-4, 0.001, 0.005, 0.01]
#         },
#         'epochs': {
#             'values': [5, 10, 15, 20]
#         },
#         'hid_layers': {
#             'values': [1, 2, 3, 4]
#         },
#         'emb_size': {
#             'values': [64, 128, 256, 512]
#         },
#         'hid_size': {
#             'values': [64, 128, 256, 512]
#         },
#         'dropout': {
#             'values': [0, 0.1, 0.2, 0.3, 0.4]
#         },
#         'type_t': {
#             'values': ['RNN', 'LSTM', 'GRU']
#         }
#     }
# }


# #sample wandb run()
# def run():
#   config_defaults = {
#         'optimizer': 'Adam',
#         'learning_rate': 0.005,
#         'epochs': 1,
#         'hid_layers': 1,
#         'emb_size': 256,
#         'hid_size': 256,
#         'dropout': 0.1,
#         'type_t': 'GRU'
#   }
#   wandb.init(config=config_defaults)
#   config = wandb.config
#   encoder = Encoder(config.type_t, inp_lang.n_letters, config.emb_size, config.hid_size, config.dropout, config.hid_layers).to(device)
#   decoder = Decoder(config.type_t, out_lang.n_letters, config.emb_size, config.hid_size, config.dropout, config.hid_layers).to(device)
#   train = Train(train_pairs, encoder, decoder, nn.NLLLoss(), config.dropout)
#   train.trainIters(config.optimizer, config.learning_rate,print_every= 10000, epochs=config.epochs)

#   wandb.finish()

# sweep_id = wandb.sweep(sweep_config,project='assignment-3')
# wandb.agent(sweep_id, function=run, count=1)


# In[ ]:


# train_pairs, val_pairs,test_pairs, inp_lang, out_lang = readLang('eng', 'hin')

# encoder = Encoder('GRU', inp_lang.n_letters, 512, 512, 0, 1).to(device)
# decoder = Decoder('GRU', out_lang.n_letters, 512, 0, 1).to(device)
# train = Train(train_pairs, encoder, decoder, nn.NLLLoss())
# train.train_Iters('SGD', 0.01, print_every=1000, epochs=10)
# data = test_pairs
# Test_accuracy = train.evaluateData(data)

