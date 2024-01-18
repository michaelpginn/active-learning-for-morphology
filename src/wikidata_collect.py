#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:25:43 2023

@author: salkazzar
"""

import pandas as pd
import re
from collections import Counter
import string
import random

#%%#
def wiki_file(lang):
    language = lang
    wiki_dump = open(language+'wiki-20230220-pages-articles-multistream.txt', 'r') 
    wiki_lines= wiki_dump.readlines()
    return wiki_lines

def word_split(text,n):
    # split on period, exclamation or question mark
    sentence = re.split(r'\.|!|\?', str(text))
    print("No. of sentences: " + str(len(sentence)))
    #sample n sentences
    sentence_sample=sentence[0:n]
    #Remove all punctuation and split based on white-space
    translator = str.maketrans('', '', string.punctuation)
    sent_clean = [s.translate(translator) for s in sentence_sample]
    #Split sentences in to words
    words =[word for line in sent_clean for word in line.split()]    
    return words

def unimorph_compare(words,lang):
    #calculate frequency of words in sentences sampled
    freq = pd.DataFrame(Counter(words).items())
    freq.columns = ["wordform", "freq"]
    print("No. of Wiki words: " + str(len(freq.wordform)))
    #Match to wordform to unimorph database
    uni_path = "unimorph/"
    lang_path = lang
    data = pd.read_csv(uni_path + lang_path , sep="\t", header=None)
    data.columns = ["lemma", "wordform", "MSD"]
    # A n B, find the intersect between unimorph database and wikidata
    intersect = data[data['wordform'].isin(freq["wordform"])]
    # A-B, find difference between unimorph and wikidata. Since we don't have the mapping of wordform to lemma for wikidata (B), calculate A - AnB instead)
    #This step is for test cases. Don't want lemma overlap. 
    difference = data[~data['lemma'].isin(intersect["lemma"])]
    print("No. of Unimorph forms: " + str(len(difference.wordform)))
    #Sort based on frequency
    intersect = intersect.merge(freq[['wordform', 'freq']], how='left', on='wordform')
    sorted_intersect = intersect.sort_values(by=['freq'],ascending = False)
    print("No. of intersect: " + str(len(intersect.wordform)))
    #Get unimorph lemma dictionary
    lemma_dict = data.lemma.unique().tolist()
    # Get Unique Count of MSD and lemma 
    MSD_count = intersect.MSD.nunique()
    lemma_count = intersect.lemma.nunique()
    print("Unique MSD count: "+ str(MSD_count))
    print("Unique lemma count: "+ str(lemma_count))
    return sorted_intersect, difference, lemma_dict

def uni_dict(lemma_list):
    copy = ['COPY']
    num =len(lemma_list)
    c_tag = num*copy
    lemma_dict=pd.DataFrame(list(zip(lemma_list, lemma_list,c_tag)), columns =['lemma', 'wordform','MSD'])
    return lemma_dict

def lemma_coverage(intersect):
    #List unique lexemes in wiki words data
    lex = intersect.lemma.unique().tolist()
    #count number of forms per lemma
    forms = []
    for l in range(len(lex)):
        forms.append(len(intersect.loc[intersect['lemma'] == lex[l]]))
    #create new df 'lemma','#forms'
    lem_cov = pd.DataFrame(list(zip(lex,forms)),columns=['lemma','no.forms'])
    return lem_cov 
   
def dev_sample(sorted_df,sample_size):
    #sample p most frequent wordform
    dsample = sorted_df.sample(sample_size)
    #create remaining sample pool
    dsample_remaining =  sorted_df.drop(dsample.index)
    return dsample, dsample_remaining


def train_sample(sorted_df,sample_size):
    #sample p most frequent wordform
    sample = sorted_df.head(sample_size)
    #create remaining sample pool
    sample_remaining = sorted_df.tail(sorted_df.shape[0] -sample_size)
    return sample, sample_remaining

def lemma_sample(difference_df, lex_num):
    #Separate nouns and adjectives 
    adj = difference_df[difference_df['MSD'].str.contains("ADJ;")]
    noun = difference_df[difference_df['MSD'].str.contains("N;")]    
    #List unique lexemes in unimorph only data
    A_lex = adj.lemma.unique().tolist()
    N_lex= noun.lemma.unique().tolist()
    #Randomly shufftle lexemes 
    A_lex_rand = random.sample(A_lex, len(A_lex))
    N_lex_rand = random.sample(N_lex, len(N_lex))
    A_tables = pd.DataFrame()
    N_tables= pd.DataFrame()
    if not adj.empty:
        #Select half of lex_num as adj & other half as N 
        A  = A_lex_rand[0:3]
        N = N_lex_rand[0:3] 
        for k in range(len(A)):
            A_tables = A_tables.append(adj[adj['lemma'].str.contains(A[k])])
        for h in range(len(N)):
            N_tables= N_tables.append(noun[noun['lemma'].str.contains(N[h])])  
    else:
        N_only = N_lex_rand[0:(lex_num)] 
        for g in range(len(N_only)):
            N_tables = N_tables.append(noun[noun['lemma'].str.contains(N_only[g])])
    #create remaining sample pool
    UM_diff = pd.concat([A_tables, N_tables])
    UM= pd.concat([difference_df,UM_diff]).drop_duplicates(keep=False)
    return A_tables, N_tables, UM
    
    
def test_sample(difference_df, lex_num):
    #List unique lexemes in unimorph only data
    lex = difference_df.lemma.unique().tolist()
    #Randomly select lexemes (Should we try to get even distribution over parts of speech? I.e. equal amounts of verbs, nouns & adjectives)
    lex_rand = random.sample(lex, len(lex))
    #Select a certain amount of lexemes
    test_lex = lex_rand[0:lex_num]
    #Extract full tables for these lexemes
    tables =pd.DataFrame()
    for k in range(len(test_lex)):
        tables = tables.append(difference_df[difference_df['lemma'].str.contains(test_lex[k])])
    #create remaining sample pool
    UMp =  pd.concat([difference_df,tables]).drop_duplicates(keep=False)
    print(tables)
    return test_lex, tables, UMp

def fairseq_format(sample_df):
    # Format according to fairseq requirements (introduce space between characters and '#')
    lemma_sp = []
    wordform_sp = []
    for i in sample_df['lemma']:
        a = (" ".join(i))
        lemma_sp.append(a + " #")
    for j in sample_df['wordform']:
        b = (" ".join(j))
        wordform_sp.append(b)  
    input_source =  [i + j for i, j in zip(lemma_sp, sample_df["MSD"])]
    return input_source, wordform_sp

def write_file(fname,examples):
    name=fname
    output=open(name,'w')
    for element in examples:
        output.write(element)
        output.write('\n')
    output.close()
