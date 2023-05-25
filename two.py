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

    def add_letter(self, letter): # making a dictionary of letters and their counts
        if letter not in self.letter_to_index:
            self.letter_to_index[letter] = self.n_letters
            self.letter_to_count[letter] = 1
            self.index_to_letter[self.n_letters] = letter
            self.n_letters = self.n_letters + 1
        else:
            self.letter_to_count[letter] = self.letter_to_count[letter] + 1

    def add_word(self, letter): # adding a word to the dictionary
        for letter in letter:
            self.add_letter(letter)

  #  def decode(self, target):
  #    return ' 'join([self.index_to_letter[i.get] for i in target])

    def decode(self, target):
        words = []
        for i in target:
            words.append(self.index_to_letter[i.get])
        return ' '.join(words)


# In[ ]:


def readLang(lang1, lang2, reverse=False): # read the file and make a dictionary of words of both languages

    # Read the file and split into lines
    train_lines = open('/home/user/Desktop/Dharani/hin_train (1).csv', encoding='utf-8').\
        read().strip().split('\n')
    val_lines = open('/home/user/Desktop/Dharani/hin_valid (1).csv', encoding='utf-8').\
        read().strip().split('\n')
    test_lines = open('/home/user/Desktop/Dharani/hin_test (1).csv', encoding='utf-8').\
        read().strip().split('\n')

   # Split every line into pairs and normalize
    train_pairs = []
    for l in train_lines:
        train_pairs.append(l.split(','))

    val_pairs = []
    for l in val_lines:
        val_pairs.append(l.split(','))
        
    test_pairs = []
    for l in test_lines:
        test_pairs.append(l.split(','))
 


    inp_lang = Lang(lang1)
    out_lang = Lang(lang2)

    for pair in train_pairs:
        inp_lang.add_word(pair[0])
        out_lang.add_word(pair[1])
    
    for pair in val_pairs:
        inp_lang.add_word(pair[0])
        out_lang.add_word(pair[1])
        
    for pair in test_pairs:
        inp_lang.add_word(pair[0])
        out_lang.add_word(pair[1])
            
   

    return train_pairs, val_pairs, test_pairs, inp_lang, out_lang


# In[ ]:


def indexes_From_word(lang, word): # convert a word to a list of indexes
    indexes_ = []
    for letter in word:
        indexes_.append(lang.letter_to_index[letter])
    return indexes_


def tensor_From_word(lang, word): # convert a word to a tensor
    indexes = indexes_From_word(lang, word)
    indexes.append(eos)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_From_Pair(pair, inp_lang, out_lang): # convert a pair of words to a pair of tensors
    inp_tensor = tensor_From_word(inp_lang, pair[0])
    Target_tensor = tensor_From_word(out_lang, pair[1])
    return (inp_tensor, Target_tensor)


# In[ ]:


class Encoder(nn.Module): # encoder class
    def __init__(self,rnn_type, inp_size, emb_size, Target_tensor, p, num_layers):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.Target_tensor = Target_tensor
        self.num_layers = num_layers
        self.embedding = nn.Embedding(inp_size, emb_size)
        self.rnn = nn.RNN(emb_size, Target_tensor, num_layers, dropout = p)
        self.gru = nn.GRU(emb_size, Target_tensor, num_layers, dropout = p)
        self.lstm = nn.LSTM(emb_size, Target_tensor, num_layers, dropout = p)
        self.rnn_type = rnn_type

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) #embedding of word
        embedded = self.dropout(embedded)
        output = embedded
        
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
                torch.zeros(self.num_layers, 1, self.Target_tensor, device=device),
                torch.zeros(self.num_layers, 1, self.Target_tensor, device=device),
            )
        return torch.zeros(self.num_layers, 1, self.Target_tensor, device=device)


# In[ ]:


class Att_Decoder(nn.Module): # decoder class
    def __init__(self, rnn_type, out_size, Target_tensor, p, num_layers):
        super(Att_Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.out_size = out_size
        self.Target_tensor = Target_tensor
        self.num_layers = num_layers
        self.embedding = nn.Embedding(out_size, Target_tensor)
        self.attn = nn.Linear(Target_tensor*2, 50)
        self.attn_combine = nn.Linear(Target_tensor*2, Target_tensor)
        self.rnn = nn.RNN(Target_tensor, Target_tensor, num_layers, dropout = p)
        self.gru = nn.GRU(Target_tensor, Target_tensor, num_layers, dropout = p)
        self.lstm = nn.LSTM(Target_tensor, Target_tensor, num_layers, dropout = p)
        self.out = nn.Linear(Target_tensor, out_size)
        self.rnn_type = rnn_type

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) # embedding of word
        embedded = self.dropout(embedded)
        Att_Weight = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) # attention weights
        Att_Applied = torch.bmm(Att_Weight.unsqueeze(0), encoder_outputs.unsqueeze(0)) # attention applied
        output = torch.cat((embedded[0], Att_Applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        # giving output according to model type
        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(output, hidden)
        elif self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        elif self.rnn_type == 'LSTM':
            output, hidden = self.lstm(output, hidden)

        # softmaxfunction to get probabilities
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, Att_Weight
    
    def initHidden(self): # initializing hidden layer
        if self.v == 'LSTM':
            return (torch.zeros(self.num_layers, 1, self.Target_tensor, device=device), 
                    torch.zeros(self.num_layers, 1, self.Target_tensor, device=device))
        return torch.zeros(self.num_layers, 1, self.Target_tensor, device=device)


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
        self.train_pairs, self.val_pairs, self.test_pairs, self.inp_lang, self.out_lang = readLang('eng', 'hin')
        #self.training_pairs = [tensors_From_Pair(self.train_pairs[i], self.inp_lang, self.out_lang) for i in range(len(self.train_pairs))]
        self.training_pairs = []
        for i in range(len(self.train_pairs)):
            pair = self.train_pairs[i]
            tensors = tensors_From_Pair(pair, self.inp_lang, self.out_lang)
            self.training_pairs.append(tensors)

    def train(self, inp_tensor, Target_tensor, encoder_optimizer, decoder_optimizer):
        encoder_hidden = self.encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs = torch.zeros(50, self.encoder.Target_tensor, device=device)

        loss = 0

        input_length = inp_tensor.size(0)
        target_len = Target_tensor.size(0)

        for i in range(input_length): # encoding a word
            
            encoder_output, encoder_hidden = self.encoder(inp_tensor[i], encoder_hidden)
            # print(encoder_output.shape)
            # print(encoder_output.shape)
            # print(encoder_hid.shape)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[sos]], device=device)
        decoder_hidden = encoder_hidden # encoder shares its hidden layer with decoder_

        Teacher_Forcing = True if random.random() < self.tf_ratio else False
        
        if Teacher_Forcing: 
            for i in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, Target_tensor[i])
                decoder_input = Target_tensor[i] # teacher _forcing

        else:
            for i in range(target_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1) # top_ k predictions
                decoder_input = topi.squeeze().detach() # detach from history as input_
                loss += self.criterion(decoder_output, Target_tensor[i])
                if decoder_input.item() == eos: # if EOS token is predicted_, stop
                    break

        loss.backward() 
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_len


    def train_Iters(self, optimizer, learning_rate, n_iters = 69, print_every = 69, epochs=-1):
        start = time.time()
        print_loss_total = 0

        if optimizer == 'SGD':
            encoder_optimizer = optim.SGD(self.encoder.parameters(), lr = learning_rate)
            decoder_optimizer = optim.SGD(self.decoder.parameters(), lr = learning_rate)
        elif optimizer == 'Adam':
            encoder_optimizer = optim.Adam(self.encoder.parameters(), lr = learning_rate)
            decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = learning_rate)
        elif optimizer == 'RMSprop':
            encoder_optimizer = optim.RMSprop(self.encoder.parameters(), lr = learning_rate)
            decoder_optimizer = optim.RMSprop(self.decoder.parameters(), lr = learning_rate)
        elif optimizer == 'NAdam':
            encoder_optimizer = optim.NAdam(self.encoder.parameters(), lr = learning_rate)
            decoder_optimizer = optim.NAdam(self.decoder.parameters(), lr = learning_rate)

        if epochs != -1: # if epochs are specified
            n_iters = len(self.train_pairs)
        else:
            train_loss_total = 0
            for iter in tqdm(range(1, n_iters+1)):
                training_pair = self.training_pairs[iter - 1]
                inp_tensor = training_pair[0]
                Target_tensor = training_pair[1]
                loss = self.train(inp_tensor, Target_tensor, encoder_optimizer, decoder_optimizer)
                train_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
            train_acc = self.evaluateData(self.train_pairs) #evaluating the model on train pairs_
            valid_acc = self.evaluateData(self.val_pairs) # evaluating the model on validation pairs_
            #test_acc = self.evaluateData(self.test_pairs)
            return train_acc, valid_acc

        Training_loss = []
        Val_accuracy = []
        Training_accuracy = []
        for j in range(epochs):
            train_loss_total = 0
            for iter in tqdm(range(1, n_iters+1)):
                training_pair = self.training_pairs[iter - 1]
                inp_tensor = training_pair[0]
                Target_tensor = training_pair[1]
                loss = self.train(inp_tensor, Target_tensor, encoder_optimizer, decoder_optimizer)
                train_loss_total += loss
                print_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
            train_acc = self.evaluateData(self.train_pairs)
            valid_acc = self.evaluateData(self.val_pairs)
            Training_loss.append(train_loss_total / n_iters)
            Val_accuracy.append(valid_acc)
            Training_accuracy.append(train_acc)
            print({'train_loss': train_loss_total / n_iters, 'train_acc': train_acc, 'valid_acc': valid_acc})
            wandb.log({'train_loss': train_loss_total / n_iters, 'train_acc': train_acc, 'valid_acc': valid_acc})
        return Training_loss, Training_accuracy, Val_accuracy
                    

    def evaluate(self, word):
        with torch.no_grad():
            inp_tensor = tensor_From_word(self.inp_lang, word)
            input_length = inp_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(50, self.encoder.Target_tensor, device=device)

            for i in range(input_length): # encoding_ a word
                encoder_output, encoder_hidden = self.encoder(inp_tensor[i], encoder_hidden)
                # print(encoder_output.shape)
                encoder_outputs[i] += encoder_output[0, 0]

            decoder_input = torch.tensor([[sos]], device=device)
            decoder_hidden = encoder_hidden # encoder_ shares its hidden layer with decoder

            decoded_word = ''
            decoder_attentions = torch.zeros(50, 50)

            for j in range(50):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[j] = decoder_attention.data
                topv, topi = decoder_output.topk(1) # top_ k predictions
                if topi.item() == eos:
                    break
                else:
                    decoded_word += (self.out_lang.index_to_letter[topi.item()])
                decoder_input = topi.squeeze().detach() # detach_ from history as input

            return decoded_word, decoder_attentions[:j+1]
        
    def evaluateData(self, data):
        acc = 0
        for word,target in data:
            output_word, attentions = self.evaluate(word)
            acc += (output_word == target)
        return acc / len(data)
            


# In[ ]:


train_pairs, val_pairs,test_pairs, inp_lang, out_lang = readLang('eng', 'hin')


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
    decoder = Att_Decoder(config.type_t, out_lang.n_letters, config.emb_size, config.hid_size, config.dropout, config.hid_layers).to(device)
    train = Train(train_pairs, encoder, decoder, nn.NLLLoss(), config.dropout)
    train.trainIters(config.optimizer, config.learning_rate, print_every=10000, epochs=config.epochs)
    wandb.finish()

    
if __name__ == '__main__':
    args = parse_args()
    run(args)


# In[ ]:





# In[ ]:


# sweep_config = {
#     'method': 'random', 
#     'metric': {
#         'name': 'valid_acc',
#         'goal': 'maximize' # goal is to maximize the validation accuracy
#     },
#     'parameters': {
#         'optimizer': {
#             'values': ['SGD', 'Adam', 'RMSprop']
#         },
#         'learning_rate': {
#             'values': [1e-4, 5e-4, 0.001, 0.005]
#         },
#         'epochs': {
#             'values': [10]
#         },
#         'hid_layers': {
#             'values': [1]
#         },
#         'emb_size': {
#             'values': [64, 128, 256, 512]
#         },
#         'Target_tensor': {
#             'values': [64, 128, 256, 512]
#         },
#         'dropout': {
#             'values': [0, 0.1, 0.2, 0.3]
#         },
#         'type_t': {
#             'values': ['RNN', 'LSTM', 'GRU']
#         }
#     }
# }



# def run():
#     # Default values for hyper-parameters
#     config_defaults = {
#         'optimizer': 'Adam',
#         'learning_rate': 0.005,
#         'epochs': 10,
#         'hid_layers': 1,
#         'emb_size': 256,
#         'Target_tensor': 256,
#         'dropout': 0.1,
#         'type_t': 'GRU'
#     }
#     wandb.init(config=config_defaults) # Initialize a new wandb run
#     config = wandb.config # config saves hyperparameters and inputs
#     encoder = Encoder(config.type_t, inp_lang.n_letters, config.emb_size, config.Target_tensor, config.dropout, config.hid_layers).to(device)
#     decoder = Att_Decoder(config.type_t, out_lang.n_letters, config.Target_tensor, config.dropout, config.hid_layers).to(device)
#     train = Train(train_pairs, encoder, decoder, nn.NLLLoss())
#     train.train_Iters(config.optimizer, config.learning_rate,print_every= 1000, epochs=config.epochs)

#     wandb.finish()




# sweep_id = wandb.sweep(sweep_config, project='assignment-3-attention')
# wandb.agent(sweep_id, function=run, count=10)


# In[ ]:


# train_pairs, val_pairs,test_pairs, inp_lang, out_lang = readLang('eng', 'hin')

# encoder = Encoder('GRU', inp_lang.n_letters, 512, 512, 0, 1).to(device)
# decoder = Att_Decoder('GRU', out_lang.n_letters, 512, 0, 1).to(device)
# train = Train(train_pairs, encoder, decoder, nn.NLLLoss())
# train.train_Iters('SGD', 0.01, print_every=1000, epochs=10)
# data = test_pairs
# Test_accuracy = train.evaluateData(data)

