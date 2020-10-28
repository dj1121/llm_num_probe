# -----------------------------------------------------------
# Functions for outputting results visually.
#
# Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
# University of Washington Linguistics (2020)
# Contact Email: dj1121@uw.edu
# -----------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_lc(out, exp_id):
    """
    Plot learning curves of training results.

    Parameters:
        - out (str): Path to out folder
        - exp_id (str): Unique identifier of this experiment

    Return:
        - None
    """
    
    df = pd.read_csv(out + exp_id + "_out.csv")
    epochs = df['epoch']
    losses = df['loss']
    train_acc = df['train_acc']
    val_acc = df['val_acc']
    
    ##########
    #  Loss  #
    ##########
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(np.arange(len(epochs)), losses, '.-')
    plt.xticks(np.arange(0, 20))
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Per Epoch \n(" + exp_id + ")")
    # plt.show()
    plt.savefig(out + exp_id + "_loss.png", dpi=400)

    ###############
    #  Train_Acc  #
    ###############
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(np.arange(len(epochs)), train_acc, '.-')
    plt.xticks(np.arange(0, 20))
    plt.yticks((np.arange(0, 1.1, 0.1)))
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Train Accuracy", fontsize=12)
    plt.title("Train Accuracy Per Epoch \n(" + exp_id + ")")
    # plt.show()
    plt.savefig(out + exp_id + "_train.png", dpi=400)


    #############
    #  Val_Acc  #
    #############
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(np.arange(len(epochs)), val_acc, '.-')
    plt.yticks((np.arange(0, 1.1, 0.1)))
    plt.xticks(np.arange(0, 20))
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title("Validation Accuracy Per Epoch \n(" + exp_id + ")")
    # plt.show()
    plt.savefig(out + exp_id + "_val.png", dpi=400)
