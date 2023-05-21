# ASSIGNMENT_3
# Character-level Neural Machine Transliteration (Seq2Seq) with Attention

This repository contains an implementation of a character-level neural machine transliteration model with attention mechanism using PyTorch. The model is designed to transliterate English words into Hindi. 

## Overview

The transliteration model consists of an encoder-decoder architecture with and without attention mechanism. The encoder processes the input English sequence, while the decoder generates the corresponding Hindi transliteration character by character. The attention mechanism allows the decoder to focus on different parts of the input sequence during the generation process.
link to wandb report ()

## Dataset
the [Aksharantar](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view?pli=1) dataset released by AI4Bharat was used in the project.This dataset contains pairs of the following form: 
xx x,yy y
ajanabee,अजनबी
i.e., a word in the native script and its corresponding transliteration in the Latin script (how we type while chatting with our friends on WhatsApp etc). Given many such (xi,yi)i=1n(x_i, y_i)_{i=1}^n
(xi
​
,yi
​
)i=1
n
​

 pairs your goal is to train a model y=f^(x)y = \hat{f}(x)
y=f
^
​
(x)
 which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर).Refer this [blog](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) for pytorch for understanding 


## building a RNN model

### 1.System Requirements:

To run the RNN model for neural machine transliteration, the system requirements are as follows:

1.1 Python: Make sure you have Python 3.x installed on your system.

1.2 PyTorch: The model implementation utilizes the PyTorch library for building and training the RNN models. Install PyTorch by following the instructions provided on the official PyTorch website (https://pytorch.org)    based on your system configuration.

1.3 Dependencies: Install the required dependencies by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

1.4 Dataset: Download the English-Hindi transliteration dataset and place it in the specified data directory. Make sure the dataset is appropriately preprocessed before training the model.

### 2.Train the model:

for loss and accuracy for training and validation datasets wandb framework is used.To findout the best hyperparameter bayesian search is employed for sweeps. The sweep configuration and default configurations of hyperparameters are specficied as follows:

   ```bash
   sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'valid_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'optimizer': {
            'values': ['SGD', 'Adam', 'RMSprop', 'NAdam']
        },
        'learning_rate': {
            'values': [1e-4, 5e-4, 0.001, 0.005, 0.01]
        },
        'epochs': {
            'values': [5, 10, 15, 20]
        },
        'hid_layers': {
            'values': [1, 2, 3, 4]
        },
        'emb_size': {
            'values': [64, 128, 256, 512]
        },
        'hidden_size': {
            'values': [64, 128, 256, 512]
        },
        'dropout': {
            'values': [0, 0.1, 0.2, 0.3, 0.4]
        },
        'type_t': {
            'values': ['RNN', 'LSTM', 'GRU']
        }
    }
```

### 3.Hyperparameter sweeps

Two self-contained Colab notebooks are provided for running the RNN model for neural machine transliteration. These notebooks are designed to be executed on a GPU-based 'CUDA' runtime session in Colab. The notebooks include all the necessary code and configurations, and the results will be logged automatically to the user's wandb account. Before starting the run, the user needs to update their wandb account details in the notebook. This allows for easy tracking and visualization of training progress and evaluation metrics using the wandb platform. Simply open the notebooks in Colab, update the wandb account information, and run the cells to train and evaluate the RNN model for transliteration. The notebooks are optimized for GPU acceleration and provide a convenient and interactive environment for running the model.

## Results

The model achieves an accuracy of X% on the validation set and demonstrates good performance in transliterating English words to Hindi.

## References

[PYTORCH-blog](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) for understanding to build neural sequence-to-sequence models.



