# -----------------------------------------------------------
# Functions for training/testing a specified transformer model 
# for experiment and output results.
#
# Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
# University of Washington Linguistics (2020)
# Contact Email: dj1121@uw.edu
# -----------------------------------------------------------

import pandas as pd
import argparse
import numpy as np
import json
import time
import sys
import re
import os
import data_handling
import visualize
import torch
import models

TIME = time.strftime("%Y%m%d-%H%M%S")
SAVE_MODEL_PATH = "./../saved_models/"

def parse_args():
    """
    Parse all command line arguments

    Parameters:
        - None

    Returns:
        - args (argparse.Namespace): The list of arguments passed in
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",type=str, help = "Whether to train or test [test,train]", default = "train")
    parser.add_argument("-task",type=str, help = "Whether to train/test on task 1 or task 2 [t1,t2]", default = "t1")
    parser.add_argument("-sents_bare",type=str, help = "Whether to use sentences or bare numbers [sents,bare]", default = "bare")
    parser.add_argument("-data_dir",type=str, help = "Path to input data files", default = "./../data/")
    parser.add_argument("-out",type=str,  help = "Path to output files", default = "./../results/")
    parser.add_argument("-saved",type=str, help = "Path to saved model if testing", default = None)
    parser.add_argument("-model",type=str,  help = "Model to train [baseline,bert,xlm,d-bert]", default = "bert")
    parser.add_argument("-lang",type=str, help = "Language of experiment [en,ja,fr,dk]", default = "en")
    parser.add_argument("-lr",type=float, help="The learning rate",default=0.01)
    parser.add_argument("-epochs",type=int,help="The number of training epochs (int)",default=20)
    parser.add_argument("-mb",type=int,help="Minibatch size",default=32)
    parser.add_argument('-fine_tune', help='Whether or not to fine tune the model', default=False)
    args = parser.parse_args()

    if args.mode == "test" and args.saved == None:
        raise Exception("If testing a model, must specify the saved model location using -saved_model [location]")
    
    return args

if __name__ == '__main__':

    args = parse_args()
    
    # Make experiment identifier/folders
    EXP_ID = TIME + "_" + args.model + "_" + args.task + "_" + args.sents_bare + "_" + args.lang
    data_path_exp = args.data_dir + "/" + args.task + "/" + args.lang  + "/" + args.sents_bare + "/"
    out_path_exp = args.out + "/" + EXP_ID + "/"
    if not os.path.exists(SAVE_MODEL_PATH): 
        os.makedirs(SAVE_MODEL_PATH)
    if not os.path.exists(data_path_exp): 
        os.makedirs(data_path_exp)
    if not os.path.exists(out_path_exp): 
        os.makedirs(out_path_exp)

    # Start testing/training
    if args.mode == "test":
        print("Testing Mode")
        print('Loading Model')
        model, tokenizer = models.load_model(args.model, args.fine_tune, saved=args.saved)

        print('Loading Data')
        _, test_set, __ = data_handling.load(data_path_exp, args.mb, tokenizer, DEVICE.type != "cpu")

        print("Testing Data")
        acc = float(eval(test_set, model))
        print("Accuracy on Test:", acc)

    elif args.mode == "train":
        print('Training Mode')
        print('Loading Model')
        model, tokenizer = models.load_model(args.model, args.fine_tune, saved=args.saved)

        print('Loading Data')
        train_set, test_set, val_set = data_handling.load(data_path_exp, args.mb, tokenizer, DEVICE.type != "cpu")

        print("Starting Training")
        model = models.train(args.lr, train_set, val_set, args.epochs, model, out_path_exp, EXP_ID)
        print('Finished Training\nSaving Model')
        models.save_model(model, EXP_ID)

        print("Saving Learning Curves")
        visualize.plot_lc(out_path_exp, EXP_ID)
