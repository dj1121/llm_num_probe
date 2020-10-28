# -----------------------------------------------------------
# Functions for dealing with models and model classes.
#
# Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
# University of Washington Linguistics (2020)
# Contact Email: dj1121@uw.edu
# -----------------------------------------------------------

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

SAVE_MODEL_PATH = "./../saved_models/"
if torch.cuda.is_available():
    print('Using Cuda')
    DEVICE = torch.device("cuda")
else:
    print('Using CPU')
    DEVICE = torch.device("cpu")

def save_model(model, exp_id):
    """
    Save trained model to disk.

    Parameters:
        - model: Model to train
        - exp_id (str): Identifier for experiment

    Returns:
        - None
    """
    torch.save(model.state_dict(), SAVE_MODEL_PATH + exp_id)

def load_model(model_name, fine_tune, saved):
    """
    Load model for training or testing.

    Parameters:
        - model_name (str): Model identifier for our experiment OR path if we are loading saved model
        - fine_tune (bool): Whether we want to fine tune the model
        - saved (str): Path for loading saved model

    Returns:
        - model (...ForSequenceClassifcation) Sequence classification transformer model 
        - tokenizer (transformers.tokenization_...) Tokenizer schema of given model (if transformer)
    """
        
    if model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMForSequenceClassification.from_pretrained('xlm-mlm-100-1280')

    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

    elif model_name == 'd-bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')
    
    elif model_name == 'word2vec':
        print()

    if saved != None:
        model.load_state_dict(torch.load(saved))

    if not fine_tune:
        for name, param in model.named_parameters():
            if 'classifier' not in name: # classifier layer
                # TODO: Fix for XLM?
                param.requires_grad = False

    return model, tokenizer

def train(lr, train_set, val_set, epochs, model, out, exp_id):
    """
    Train a model using the specified parameters

    Parameters:
        - lr (float): Learning rate
        - train_set (DataLoader): Training set
        - val_set (DataLoader): Val set
        - epochs (int): Number of epochs to train
        - model (...ForSequenceClassifcation): Model to train
        - out (str): Path to output training stats
        - exp_id (str): Unique identifier for experiment

    Returns:
        - model (...ForSequenceClassifcation): Trained model
    """

    with open(out + exp_id + "_out.csv", 'a', encoding='utf-8') as f:
            f.write("epoch,loss,train_acc,val_acc\n")

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    model = model.to(DEVICE)

    for i in range(0, epochs):
        model.train()
        for x, y in train_set:
            
            x = x.squeeze(1)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            loss, logits = model(x, labels=y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

        # Get val/train accuracy for full data each epoch
        train_acc = eval(train_set, model)  
        val_acc = eval(val_set, model) 

        # Write output csvs (train/val accuracies and loss)
        with open(out + exp_id + "_out.csv", 'a', encoding='utf-8') as f:
            f.write(str(i) + "," + str(loss.item()) + "," + str(train_acc) + "," + str(val_acc) +"\n")
        print('Epoch: {} Loss: {} Train Acc: {} Val Acc: {}'.format(i, loss.item(), train_acc, val_acc))

    return model

def eval(data, model):
    """
    Classify some data, get accuracy.

    Parameters:
        - data (DataLoader): Data to classify
        - model (...ForSequenceClassifcation): Model to classify with

    Returns:
        - acc (float): Accuracy achieved on data

    """
    model.eval()
    
    for x,y in data:
        
        x = x.squeeze(1)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        loss, logits = model(x, labels=y)
        _, predicted = torch.max(logits.detach(), 1)

    correct = (predicted == y).float().sum()
    acc = correct/x.shape[0]

    return acc.item()
