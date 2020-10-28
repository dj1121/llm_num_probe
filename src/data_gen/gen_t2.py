# -----------------------------------------------------------
# Generate data for task 2 (value comparison).
# Generates grammatical number pairs to insert into sentences or
# comapre bare.
#
# Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
# University of Washington Linguistics (2020)
# Contact Email: dj1121@uw.edu
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split
from num2words import num2words
import pandas as pd
import argparse
import random
import os
import re

def gen_pairs(r, s, lang):
    """
    Create of list of s pairs of number words (greater/lesser pairs)

    Parameters:
        - r (int): Range [1,r] that numbers can fall into 
        - s (int): Number of pairs to generate
        - lang (str): What language working with

    Returns:
        - pairs (list): (list) A list of grammatical and ungrammatical words
        - labels (list): (list) Denotes relation between numbers in pairs (0 = first > second, 1 = first < second)
    """
    pairs = []
    labels = []

    while len(pairs) < s:
        rand1 = random.randint(1,r)
        rand2 = random.randint(1,r)

        pair = num2words(rand1, lang=lang) + ";" + num2words(rand2, lang=lang)

        if rand1 != rand2 and pair not in pairs:
            pairs.append(pair)
            if rand1 > rand2: 
                labels.append(0)
            else:
                labels.append(1)   

    return pairs, labels

def to_sent(pairs, lang):
    """
    Insert number words into sentence templates.
    
    Parameters:
        - pairs (list): Pairs of number words
        - lang (str): What language working with

    Returns:
        - sents (list): List of sentences with numbers inserted into them.
    """

    # Get template sentences
    with open('./sent_templates/' + lang + '_templates.txt', 'r', encoding="utf-8") as f:
        sentences = f.readlines()

    sent_pairs = []
    for pair in pairs:
        # Pick two random sentences
        s1 = random.choice(sentences).strip()
        s2 = random.choice(sentences).strip()
        
        # Insert number words from pair
        s1 = s1.replace("***", pair.split(";")[0])
        s2 = s2.replace("***", pair.split(";")[1])

        # Add pair of sentences with numbers to sentence pairs
        sent_pairs.append(s1 + ";" + s2)

    return sent_pairs

def train_test_val_split(sents_df, no_sents_df, p_train, p_test, p_val):
    """
    Split dataframes according to train/test split. Ensure both have same number word pairs
    used in them.

    Parameters:
        - data (Pandas dataframe): Dataframe of word pairs (or sents) and labels (x,y)
        - p_train (float): Percentage of train data
        - p_test (float): Percentage of test data
        - p_val (float): Percentage of validation data

    Returns:
        - train (Pandas dataframe): Dataframe of train word pairs (or sents) and labels (x,y)
        - test (Pandas dataframe):  Dataframe of test word pairs (or sents) and labels (x,y)
        - val (Pandas dataframe): Dataframe of val word pairs (or sents) and labels (x,y)
    """

    # Split sents data twice to make sents split
    x_train, x_test, y_train, y_test = train_test_split(sents_df['x'], sents_df['y'], test_size=1 - p_train)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=p_test/(p_test + p_val)) 
    sents_train = pd.DataFrame({'x':x_train, 'y':y_train})
    sents_test = pd.DataFrame({'x':x_test, 'y':y_test})
    sents_val = pd.DataFrame({'x':x_val, 'y':y_val})

    # Make no_sents split use same as sents split (just not inserted into sents)
    no_sents_train = no_sents_df.loc[sents_train.index]
    no_sents_test = no_sents_df.loc[sents_test.index]
    no_sents_val = no_sents_df.loc[sents_val.index]

    # Reset indices on all
    sents_train = sents_train.reset_index(drop=True)
    sents_test = sents_test.reset_index(drop=True)
    sents_val = sents_val.reset_index(drop=True)
    no_sents_train = no_sents_train.reset_index(drop=True)
    no_sents_test = no_sents_test.reset_index(drop=True)
    no_sents_val = no_sents_val.reset_index(drop=True)

    return sents_train, sents_test, sents_val, no_sents_train, no_sents_test, no_sents_val

def parse_args():
    """
    Parse all command line arguments

    Parameters:
        - None

    Returns:
        - args (argparse.Namespace): The list of arguments passed in
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-range",type=int,help="Max possible value of the numbers to be generated (inclusive)",default=999)
    parser.add_argument("-samples",type=int,help="The number of pairs to be generated",default=100000)
    parser.add_argument("-out",type=str,help="Output directory for number pairs generated", default="./../../data")
    parser.add_argument("-lang",type=str,help="Language of data to be generated (en,fr,dk,ja)", default="en")
    parser.add_argument("-p_test",type=float,help="Percentage (0.0-1.0) of test data", default=0.2)
    parser.add_argument("-p_val",type=float,help="Percentage (0.0-1.0) of validation data", default=0.2)
    parser.add_argument("-p_train",type=float,help="Percentage (0.0-1.0) of train data", default=0.6)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out + "/" + "t2" + "/" + args.lang + "/sents/"):
        os.makedirs(args.out + "/" + "t2" + "/" + args.lang + "/sents/")
    if not os.path.exists(args.out + "/" + "t2" + "/" + args.lang + "/bare/"):
        os.makedirs(args.out + "/" + "t2" + "/" + args.lang + "/bare/")

    # Generate pairs of number words for comparison
    word_pairs, labels = gen_pairs(args.range, args.samples, args.lang)
    sent_pairs = to_sent(word_pairs, args.lang)

    # Create data frames
    sents_df = pd.DataFrame({'x':sent_pairs, 'y':labels})
    no_sents_df = pd.DataFrame({'x':word_pairs, 'y':labels})      
        
    # Train-test-val split of dataframes
    sents_train, sents_test, sents_val, no_sents_train, no_sents_test, no_sents_val \
         = train_test_val_split(sents_df, no_sents_df, args.p_train, args.p_test, args.p_val)

    # Output to csv
    sents_train.to_csv(args.out + "/" + "t2" + "/" + args.lang + "/sents/" + "t2" + "_" + args.lang + "_sents_" + "train.csv")
    sents_test.to_csv(args.out + "/" + "t2" + "/" + args.lang + "/sents/" + "t2" + "_" + args.lang + "_sents_" + "test.csv")
    sents_val.to_csv(args.out + "/" + "t2" + "/" + args.lang + "/sents/" + "t2" + "_" + args.lang + "_sents_" + "val.csv")
    no_sents_train.to_csv(args.out + "/" + "t2" + "/" + args.lang + "/bare/" +  "t2" + "_" + args.lang + "_bare_" + "train.csv")
    no_sents_test.to_csv(args.out + "/" + "t2" + "/" + args.lang + "/bare/" +  "t2" + "_" + args.lang + "_bare_" + "test.csv")
    no_sents_val.to_csv(args.out + "/" + "t2" + "/" + args.lang + "/bare/" +  "t2" + "_" + args.lang + "_bare_" + "val.csv")
