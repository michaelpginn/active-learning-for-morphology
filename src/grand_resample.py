#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:30:56 2023

@author: salkazzar
"""

import pandas as pd
from difflib import ndiff

#%%#
def get_file(lang):
    language = lang
    src = open('um.'+language+'-base.input', 'r') 
    src_lines= src.readlines()
    trg = open('um.'+language+'-base.output', 'r') 
    trg_lines= trg.readlines()    
    pred = open('um.'+language+'-base.guesses', 'r') 
    pred_lines= pred.readlines()    
    log = open('um.'+language+'-basen2.guesses', 'r') 
    log_lines= log.readlines() 
    return src_lines, trg_lines, pred_lines, log_lines

def clean_lines(source,target,prediction):
    src_frm = []
    for i in range(len(source)):
        src_frm.append(source[i].rstrip('\n').split('\t'))
    clean_input = [source for sources in src_frm for source in sources]
    trg_frm=[]
    for i in range(len(target)):
        trg_frm.append(target[i].rstrip('\n').split('\t'))
    clean_output = [output for outputs in trg_frm for output in outputs]
    pred_frm=[]
    for i in range(len(prediction)):
        pred_frm.append(prediction[i].rstrip('\n').split('\t'))
    clean_guess = [guess for guesses in pred_frm for guess in guesses ]
    return clean_input, clean_output, clean_guess
 
def clean_log(loglikelihood):
    log_like=[]
    for i in range(len(loglikelihood)):
        log_like.append(loglikelihood[i].rstrip('\n').split('\t'))
    #split n1  n2 values
    n1 = log_like[0::2]
    n2 = log_like[1::2]
    #Get rid of nested lists    
    clean_n1 = [float(n) for n1s in n1 for n in n1s]
    clean_n2 = [float(l) for n2s in n2 for l in n2s]
    return clean_n1, clean_n2        


def rand_sample(source, output, N):
    #create dataframe from lists        
    df = pd.DataFrame(list(zip(source, output)), columns =['input', 'output'])
    # Sample N randomly
    sample_size = N 
    rand_sample = df.sample(n = sample_size)
    rand_remain = df.drop(rand_sample.index)
    return rand_sample, rand_remain

def tlc_sample(source, output, n1, N):
    #create dataframe from lists        
    gf = pd.DataFrame(list(zip(source, output, n1)), columns =['input', 'output', 'n1'])
    # Select based on loglikelihood values #
    sample_size =N
    #sort dataframe based on loglikelihood values (ascending false for low confidence forms)
    low = gf.sort_values('n1', ascending=True)
    low_sample = low.head(sample_size)
    low_remain = low.tail(gf.shape[0] -sample_size)
    return low_sample, low_remain

def lev_dist(str_1, str_2):
    distance = 0
    buffer_removed = buffer_added = 0
    for x in ndiff(str_1, str_2):
        code = x[0]
        # Code ? is ignored as it does not translate to any modification
        if code == ' ':
            distance += max(buffer_removed, buffer_added)
            buffer_removed = buffer_added = 0
        elif code == '-':
            buffer_removed += 1
        elif code == '+':
            buffer_added += 1
    distance += max(buffer_removed, buffer_added)
    return distance

def accuracy(target, prediction):
    trg = target
    pred = prediction
    correct = []
    for i in range(len(trg)):
        if trg[i] == pred[i]:
            correct.append('1')
        else:
            correct.append('0')
    false = correct.count('0')
    true = correct.count('1')
    acc = true / (true + false)
    print(acc)
    return correct

    
def inc_sample(source, target, prediction, correct, n1, n2, N):
    lev_distance = []
    for k in range(len(target)):
        lev_distance.append(lev_dist(target[k], prediction[k]))
    #create dataframe from lists        
    hf = pd.DataFrame(list(zip(source, target, prediction, correct, n1, n2, lev_distance)), columns =['input', 'output', 'guess', 'correct', 'n1','n2', 'Lev'])
    hf['n2 - n1'] = hf['n2'] - hf['n1']
    #Count instances of incorrect prediction
    false = correct.count('0')
    # Sample incorrect forms  
    sample_size = N
    if false == sample_size:
        inc_order = hf.sort_values(['correct'], ascending=[True])
        inc_sample = inc_order.head(sample_size)
        inc_remain = inc_order.tail(hf.shape[0] -sample_size)
    if false < sample_size: 
        inc_order =  hf.sort_values(['correct', 'n2 - n1'], ascending=[True, True]) 
        inc_sample = inc_order.head(sample_size)
        inc_remain = inc_order.tail(hf.shape[0] -sample_size)
    if false > sample_size:
        inc_order  =  hf.sort_values(['correct', 'Lev'], ascending=[True, False])
        inc_sample = inc_order.head(sample_size)
        inc_remain = inc_order.tail(hf.shape[0] -sample_size)
    return inc_sample, inc_remain

def write_file(lang, sample_style, frame):
    language = lang
    sample = sample_style
    colnames=['input'] 
    colnames1=['output'] 
    fin = pd.read_csv('train.'+language+'-base.input',names=colnames, header=None)
    fout = pd.read_csv('train.'+language+'-base.output',names=colnames1, header=None)
    frame_in = pd.concat([fin, frame])
    frame_out = pd.concat([fout, frame])
    frame_in['input'].to_csv('train.'+language+'-'+sample +'.input',index=False, header=False)
    frame_out['output'].to_csv('train.'+language+'-'+sample+'.output',index=False, header=False)
    #Copy dev + test files with new name
    tstin = pd.read_csv('tst.'+language+'-base.input',names=colnames, header=None)
    tstout = pd.read_csv('tst.'+language+'-base.output',names=colnames1, header=None)
    tstin['input'].to_csv('tst.'+language+'-'+sample+'.input',index=False, header=False)
    tstout['output'].to_csv('tst.'+language+'-'+sample+'.output',index=False, header=False)    
    devin = pd.read_csv('dev.'+language+'-base.input',names=colnames, header=None)
    devout = pd.read_csv('dev.'+language+'-base.output',names=colnames1, header=None)
    devin['input'].to_csv('dev.'+language+'-'+sample+'.input',index=False, header=False)
    devout['output'].to_csv('dev.'+language+'-'+sample+'.output',index=False, header=False)

def lm_files(lang, exp):
    language = lang
    experiment = exp
    lm_score = open('um.'+language+'-base.'+experiment+ '.lm.scores', 'r') 
    lm_score_lines= lm_score.readlines()
    lm_clean =[]
    for i in range(len(lm_score_lines)):
        lm_clean.append(lm_score_lines[i].rstrip('\n').split('\t'))
    #split n1  n2 values
    score= lm_clean[1::2]
    #Get rid of nested lists    
    clean_score = [p for n1s in score for p in n1s]
    return clean_score        


def lm_sample(source, output, lm_score, N):
    
    #create dataframe from lists        
    gf = pd.DataFrame(list(zip(source, output, lm_score)), columns =['input', 'output', 'lm_score'])
    # Select based on loglikelihood values #
    sample_size =N
    #sort dataframe based on loglikelihood values (ascending false for low confidence forms)
    lm = gf.sort_values('lm_score', ascending=True)
    lm_sample = lm.head(sample_size)
    lm_remain = lm.tail(gf.shape[0] -sample_size)
    return lm_sample, lm_remain

