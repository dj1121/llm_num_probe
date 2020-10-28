# -----------------------------------------------------------
# Generate data for task 1 (grammaticality judgment).
# Generates both grammatical and ungrammatical numbers as bare
# and inserted into sentence templates.
#
# Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
# University of Washington Linguistics (2020)
# Contact Email: dj1121@uw.edu
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split
from num2words import num2words
import pandas as pd
import numpy as np
import argparse
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Grammatical number words (for language currently being generated)
GRAM_NUMS = None

# Keep the list of grammatical numbers at one million
# If we accidentally make a bigger number from ungrammatical generation, it can be caught as grammatical
GRAM_NUMS_R = 1000000

def gram_num(word, lang, GRAM_NUMS_R):
    """
    Checks if number word is grammatical by checking
    list of number words in range of GRAM_NUMS_R. Slow, but
    works for our purposes. GRAM_NUMS_R ought not to be changed.

    Parameters:
        - word (str): The word that we are attemping to parse
        - lang (str): What language working with
        - GRAM_NUMS_R (int): Max value of numbers (non-inclusive)

    Returns:
        - True or False depending on whether a valid parse was found
    """
    # Generate all possible numbers in range if not done already
    global GRAM_NUMS

    if GRAM_NUMS == None:
        GRAM_NUMS = []
        for i in range(1, GRAM_NUMS_R):
            GRAM_NUMS.append(num2words(i,lang=lang))
    
    # Check if number grammatical
    return word in GRAM_NUMS

def gen_nums(r, s, lang):
    """
    Generate grammatical and ungrammatical number words.
    Make ungrammatical number words by using two grammatical words and randomly
    splicing and concatenating them together.
    
    Parameters:
        - r (int): Range [1,r] that grammatical numbers can fall into 
        - s (int): Number of words to generate (s/2 grammatical, s/2 ungrammatical)
        - lang (str): What language working with

    Returns:
        - words (list): (list) A list of grammatical and ungrammatical words
        - labels (list): (list) Denotes which words are ungrammatical (1) / grammatical (0)
    """

    words = []
    labels = []
    word_lengths = []
    
    ######################################
    # GENERATE S/2 GRAMMATICAL NUM WORDS #
    ######################################
    while len(words) < int(s/2):
        num_word = num2words(random.randint(1, r), lang=lang)
        if num_word not in words:
            words.append(num_word)
            labels.append(0)
            word_lengths.append(len(num_word))
    
    # Max length of real number words
    max_len = np.max(word_lengths)

    ##############################################
    # GENERATE REMAINING S/2 UNGRAMMATICAL NUMS #
    ##############################################
    while len(words) < s: 
        num_pair = []

        # Generate two random numbers for pair
        num_pair.append(random.randint(1, r))
        num_pair.append(random.randint(1, r))

        if lang == "ja":
             # Get two gram. words
            w1 = list(num2words(num_pair[0], lang=lang))
            w2 = list(num2words(num_pair[1], lang=lang))
        else:
            w1 = num2words(num_pair[0], lang=lang).split()
            w2 = num2words(num_pair[1], lang=lang).split()
            
        # Randomly shuffle their elements (one thousand and two -> one two thousand and)
        random.shuffle(w1)
        random.shuffle(w2)
        
        # Insert w2 somewhere randomly into w1 array (after breaking up w1)
        rand_insert = random.randint(0, len(w1) - 1)
        w1_left = w1[0:rand_insert]
        w1_right = w1[rand_insert:]

        # Final number word
        if lang == "ja":
            num_word = "".join(w1_left + w2 + w1_right)
        else:
            num_word = " ".join(w1_left + w2 + w1_right)

        # Check if it's ungrammatical and if its below max length of grammatical words
        # Also make sure it's not already in the list (avoid duplicates)
        if not gram_num(num_word, lang, r) and (len(num_word)) <= max_len and num_word not in words:
            words.append(num_word)
            labels.append(1)
            word_lengths.append(len(num_word))

    # NOTE: For debugging purposes
    # sns.scatterplot(data=np.array(word_lengths))
    # plt.show()

    return words, labels

def to_sent(words, lang):
    """
    Insert number words into sentence templates.
    
    Parameters:
        - words (list): List of ungrammatical/grammatical number words
        - lang (str): What language working with

    Returns:
        - sents (list): List of sentences with numbers inserted into them.
    """

    # Get template sentences
    with open('./sent_templates/' + lang + '_templates.txt', 'r', encoding="utf-8") as f:
        sentences = f.readlines()

    sents = []
    for word in words:
        string = random.choice(sentences).strip()
        string = string.replace("***", word)
        sents.append(string)

    return sents

def train_test_val_split(sents_df, no_sents_df, p_train, p_test, p_val):
    """
    Split dataframes according to train/test split. Ensure both have same number words
    used in them.

    Parameters:
        - data (Pandas dataframe): Dataframe of words (or sents) and labels (x,y)
        - p_train (float): Percentage of train data
        - p_test (float): Percentage of test data
        - p_val (float): Percentage of validation data

    Returns:
        - train (Pandas dataframe): Dataframe of train words (or sents) and labels (x,y)
        - test (Pandas dataframe):  Dataframe of test words (or sents) and labels (x,y)
        - val (Pandas dataframe): Dataframe of val words (or sents) and labels (x,y)
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
    parser.add_argument("-range",type=int,help="Max value of integers to be generated (inclusive)",default=999)
    parser.add_argument("-samples",type=int,help="The number of integer pairs to be generated (across train/test/val)",default=100000)
    parser.add_argument("-out",type=str,help="Parent output directory for all data generated", default="./../../data")
    parser.add_argument("-lang",type=str,help="Language of data to be generated", default="en")
    parser.add_argument("-p_test",type=float,help="Percentage (0.0-1.0) of test data", default=0.2)
    parser.add_argument("-p_val",type=float,help="Percentage (0.0-1.0) of validation data", default=0.2)
    parser.add_argument("-p_train",type=float,help="Percentage (0.0-1.0) of train data", default=0.6)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    if not os.path.exists(args.out + "/" + "t1" + "/" + args.lang + "/sents/"):
        os.makedirs(args.out + "/" + "t1" + "/" + args.lang + "/sents/")
    if not os.path.exists(args.out + "/" + "t1" + "/" + args.lang + "/bare/"):
        os.makedirs(args.out + "/" + "t1" + "/" + args.lang + "/bare/")

    # Generate numbers turn them to text/insert into sentences
    words, labels = gen_nums(args.range, args.samples, args.lang)
    sents = to_sent(words, args.lang)

    # Create data frames
    sents_df = pd.DataFrame({'x':sents, 'y':labels})
    no_sents_df = pd.DataFrame({'x':words, 'y':labels})      
        
    # Train-test-val split of dataframes
    sents_train, sents_test, sents_val, no_sents_train, no_sents_test, no_sents_val \
         = train_test_val_split(sents_df, no_sents_df, args.p_train, args.p_test, args.p_val)

    # Output to csv
    sents_train.to_csv(args.out + "/" + "t1" + "/" + args.lang + "/sents/" + "t1" + "_" + args.lang + "_sents_" + "train.csv")
    sents_test.to_csv(args.out + "/" + "t1" + "/" + args.lang + "/sents/" + "t1" + "_" + args.lang + "_sents_" + "test.csv")
    sents_val.to_csv(args.out + "/" + "t1" + "/" + args.lang + "/sents/" + "t1" + "_" + args.lang + "_sents_" + "val.csv")
    no_sents_train.to_csv(args.out + "/" + "t1" + "/" + args.lang + "/bare/" +  "t1" + "_" + args.lang + "_bare_" + "train.csv")
    no_sents_test.to_csv(args.out + "/" + "t1" + "/" + args.lang + "/bare/" +  "t1" + "_" + args.lang + "_bare_" + "test.csv")
    no_sents_val.to_csv(args.out + "/" + "t1" + "/" + args.lang + "/bare/" +  "t1" + "_" + args.lang + "_bare_" + "val.csv")