
# -----------------------------------------------------------
# Functions for data handling (input/output)
# of experimental data
#
# Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
# University of Washington Linguistics (2020)
# Contact Email: dj1121@uw.edu
# -----------------------------------------------------------

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MAX_LEN = None
TOKENIZER = None

class ExpData(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path).reset_index()
        self.len = len(self.data)
        
    def __getitem__(self, index):
        x = prepare_input(self.data.x[index])
        y = torch.tensor(int(self.data.y[index]))
        return x, y

    def __len__(self):
        return self.len

def prepare_input(x):
    """
    Convert an input into proper input for transformer (tokenize/split sentences)

    Parameters:
        - x (str) String input
    Returns:
        - (torch.tensor): Tensor of tokenized input

    """
    global MAX_LEN
    global TOKENIZER

    # Split in case task 2 (where there is a pair)
    x = x.split(';')

    # Initialize Tokens
    tokens = [TOKENIZER.cls_token]
    for word in x:
        tokens = tokens + TOKENIZER.tokenize(word)
        tokens.append(TOKENIZER.sep_token)
    token_ids = TOKENIZER.convert_tokens_to_ids(tokens)

    # Zero-pad sequence length
    while len(token_ids) < MAX_LEN:
        token_ids.append(0)

    return torch.tensor(token_ids).unsqueeze(0)

def get_padding(data_dir):
    """
    Get length of longest input x, make all other inputs padded up to that

    Parameters:
        - data_dir (str): Path to where this experiment's data resides
    Returns:
        - max_len + 2 (int): Length of longest input x (after tokenization) + 2
    """
    global TOKENIZER

    max_len = 0
    for f_name in os.listdir(data_dir):
        df = pd.read_csv(data_dir + f_name)
        for row in df['x']:
            toks = TOKENIZER.tokenize(row)
            curr_len = len(TOKENIZER.convert_tokens_to_ids(toks))
            if curr_len > max_len:
                max_len = curr_len
        
    # Account for additional tokens
    return max_len + 2

def load(data_dir, batch_size, tokenizer, gpu_available):
    """
    Load experimental data, create DataLoader/Dataset Pytorch
    objects.

    Parameters:
        - data_dir (str): Path to where this experiment's data resides
        - batch_size (int): How large minibatches should be
        - tokenizer (transformers.tokenization_...) Model tokenizer for words
        - gpu_available (bool): Whether GPU is available
    Returns:
        - train (DataLoader)
        - test (DataLoader)
        - val (DataLoader)
    """
    global MAX_LEN
    global TOKENIZER

    # Find how much should pad/set tokenizer
    TOKENIZER = tokenizer
    MAX_LEN = get_padding(data_dir)
    

    train = None
    test = None
    val = None

    if gpu_available:
        params = {'batch_size': batch_size,
                        'shuffle': True,
                        'drop_last': False,
                        'num_workers': 8}
    else:
        params = {'batch_size': batch_size,
                        'shuffle': True,
                        'drop_last': False,
                        'num_workers': 0}

    for f_name in os.listdir(data_dir):
        if "train" in f_name:
            train = DataLoader(ExpData(data_dir + f_name), **params)            
        elif "test" in f_name:
            test = DataLoader(ExpData(data_dir + f_name), **params)
        elif "val" in f_name:
            val = DataLoader(ExpData(data_dir + f_name), **params)

    return train, test, val
