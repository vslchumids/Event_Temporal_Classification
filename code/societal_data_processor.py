################################################################
# W266 Term Project: Event Temporal State Identification
#
# John Chiang, Vincent Chu
################################################################

import nltk 
from nltk import tokenize
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

import numpy as np
import os
import logging
import sys

# Regular expression
import re

import itertools
from collections import Counter

def process_tokenized_annotated_file(tokens):
    chunk=[]
    chunk_tag=[]

    events_matrix=[]
    
    chunk_on=False
    bars_on=False

    for i in range(len(tokens)):        
        if tokens[i]=='>':
            if tokens[i-1]=='CHUNK':
                chunk_on=True
            elif tokens[i-1]=='/CHUNK':
                chunk_on=False
                chunk=chunk[:-2]
                
                for j in range(len(chunk_tag)):                    
                    events_matrix.append([chunk, chunk_tag[j]])
                    
                chunk=[]
                chunk_tag=[]
        elif chunk_on==True:
            if tokens[i]=='|||':
                bars_on=True
            elif tokens[i].find("NO=") > -1:
                bars_on=False
            elif bars_on:  
                if tokens[i-1]=='|||':
                    chunk_tag.append(tokens[i])
            else:
                chunk.append(tokens[i])
    
    return events_matrix #(events, events_tags)

def process_tokenized_annotated_file_v2(tokens):
    chunk=[]
    chunk_tag=[]

    events_matrix=[]
    
    chunk_on=False
    bars_on=False

    for i in range(len(tokens)):        
        if tokens[i]=='>':
            if tokens[i-1]=='CHUNK':
                chunk_on=True
                chunk=[]
            elif tokens[i-1]=='/CHUNK':
                chunk_on=False
                chunk=chunk[:-2]
                
                #for j in range(len(chunk_tag)):                    
                #    events_matrix.append([chunk, chunk_tag[j]])
                    
                chunk=[]
                #chunk_tag=[]
        elif chunk_on==True:
            if tokens[i]=='|||':
                bars_on=True
            elif tokens[i].find("NO=") > -1:
                bars_on=False
            elif bars_on:  
                if tokens[i-1]=='|||':
                    #chunk_tag.append(tokens[i])
                    #print "In bars_on:", chunk
                    chunk_copy = chunk[:]
                    events_matrix.append([chunk_copy, tokens[i]])                    
                    #chunk=[]
            else:
                #print tokens[i]
                chunk.append(tokens[i])
    
    return events_matrix #(events, events_tags)

def process_tokenized_annotated_file_v3(tokens):
    chunk=[]
    chunk_tag=[]

    events_matrix=[]
    
    chunk_on=False
    bars_on=False

    for i in range(len(tokens)):        
        if tokens[i]=='>':
            if tokens[i-1]=='CHUNK':
                chunk_on=True
                chunk=[]
            elif tokens[i-1]=='/CHUNK':
                chunk_on=False
                chunk=chunk[:-2]
                
                #for j in range(len(chunk_tag)):                    
                #    events_matrix.append([chunk, chunk_tag[j]])
                    
                chunk=[]
                #chunk_tag=[]
        elif chunk_on==True:
            if tokens[i]=='|||':
                bars_on=True
            elif tokens[i].find("NO=") > -1:
                bars_on=False
            elif bars_on:  
                if tokens[i-1]=='|||':
                    #chunk_tag.append(tokens[i])
                    #print "In bars_on:", chunk
                    chunk_copy = chunk[:]
                    events_matrix.append([chunk_copy, tokens[i]])                    
                    chunk=[]
            else:
                #print tokens[i]
                chunk.append(tokens[i])
    
    return events_matrix #(events, events_tags)

def process_annotated_file(file_path):
    f=open(file_path,'rU')
    raw=f.read()
    tokens=tokenize.word_tokenize(raw)
    return process_tokenized_annotated_file(tokens)

def process_annotated_file_v2(file_path):
    f=open(file_path,'rU')
    raw=f.read()
    tokens=tokenize.word_tokenize(raw)
    return process_tokenized_annotated_file_v2(tokens)

def process_annotated_file_v3(file_path):
    f=open(file_path,'rU')
    raw=f.read()
    tokens=tokenize.word_tokenize(raw)
    return process_tokenized_annotated_file_v3(tokens)

def process_annotated_files_dir(dir):
    
    events_matrix=[]
    
    for dirName, subdirList, fileList in os.walk(dir):
        for fname in fileList:
            try:
                temp_matrix=process_annotated_file(dir + fname)                

                for i in range(len(temp_matrix)):
                    temp_matrix[i].append(fname)                                            
                
                events_matrix=events_matrix+temp_matrix
            except UnicodeDecodeError:
                continue
    return events_matrix 

def process_annotated_files_dir_v2(dir):
    
    events_matrix=[]
    
    for dirName, subdirList, fileList in os.walk(dir):
        for fname in fileList:
            try:
                temp_matrix=process_annotated_file_v2(dir + fname)                

                for i in range(len(temp_matrix)):
                    temp_matrix[i].append(fname)                                            
                
                events_matrix=events_matrix+temp_matrix
            except UnicodeDecodeError:
                continue
    return events_matrix 

def process_annotated_files_dir_v3(dir):
    
    events_matrix=[]
    
    for dirName, subdirList, fileList in os.walk(dir):
        for fname in fileList:
            try:
                temp_matrix=process_annotated_file_v3(dir + fname)                

                for i in range(len(temp_matrix)):
                    temp_matrix[i].append(fname)                                            
                
                events_matrix=events_matrix+temp_matrix
            except UnicodeDecodeError:
                continue
    return events_matrix 

def process_chunk_word_level(chunk, remove_stopwords = False, replace_num = False, remove_non_alpha = False, to_lower = False, to_subwords = False):        
    ret_chunk = []    
    for word in chunk:
        if to_lower:
            word = word.lower()
        
        # If remove_stopwords not on or word is not one of the stopwords
        if not remove_stopwords or not word in set(stopwords.words("english")):
            if replace_num:
                # Replace numeric strings with the <NUM> tag 
                word = re.sub("([0-9](\s[0-9])*)+", "<NUM>", word)
            if remove_non_alpha:
                # Remove the non-alphabetical characters except 
                # "'", "(", ")", "<" and ">"
                word = re.sub("[^a-zA-Z'()<>]", "", word)    
            ret_chunk.append(word)

        if to_subwords:
            ret_chunk = transform_to_subwords(ret_chunk)
    
    return ret_chunk

#def process_chunk_subword_level(chunk, remove_stopwords = False, replace_num = False, remove_non_alpha = False, to_lower = False):
#    return transform_to_subwords(process_chunk_word_level(chunk, remove_stopwords, replace_num, remove_non_alpha, to_lower))
    
def process_chunk(chunk, remove_stopwords = False, replace_num = False, remove_non_alpha = False, to_lower = False, to_subwords = False):
    
    # Join the individual words in chunk into sentences and
    # remove the extra space before ".", "," and "'"
    chunk_str = " ".join(process_chunk_word_level(chunk, remove_stopwords, replace_num, remove_non_alpha, to_lower, to_subwords))
    
    if not remove_non_alpha:
        chunk_str = re.sub(" \.", ".", chunk_str)
        chunk_str = re.sub(" ,", ",", chunk_str)
    
    chunk_str = re.sub(" '", "'", chunk_str)

    if replace_num:
        chunk_str = re.sub("<NUM>+(\s<NUM>)*", "<NUM>", chunk_str)
    
    chunk_str = re.sub("\s+", " ", chunk_str)
    
    return chunk_str

def process_chunk_sent_level(chunk, remove_stopwords = False, replace_num = False, remove_non_alpha = False, to_lower = False, to_subwords = False, ret_word_list = False): 
    sents = tokenize.sent_tokenize(process_chunk(chunk, remove_stopwords, replace_num, remove_non_alpha, to_lower, to_subwords))
    
    if ret_word_list:
        sent_word_list = []
        for sent in sents:            
            sent_word_list.append(tokenize.word_tokenize(sent))
        return sent_word_list
    
    else:
        return sents   

# Implement the subword algorithm according to http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
def transform_to_subwords(chunk):
    #chunk = ['uni','fisher','university']
    t_chunk = []

    for t in chunk:
        vowel_pos = [pos for pos, char in enumerate(t) if char in 'aeiou']
        #rules for words with 1 vowel
        if len(vowel_pos) <= 1:
            subword1 = t
            t_chunk.append(subword1)
        else:
        #rules for words with 2 vowels
            if len(vowel_pos) == 2:
                if vowel_pos[0] == 0:
                    subword1 = t[0:vowel_pos[1]+1]
                    subword2 = t[vowel_pos[1]+1:]
                    if len(subword2) < 2:
                        t_chunk.append(subword1 + subword2)
                    else:
                        t_chunk.append(subword1)
                        t_chunk.append(subword2)
                else:                    
                    subword1 = t[0:vowel_pos[0]+1]
                    subword2 = t[vowel_pos[0]+1:vowel_pos[1]+1]
                    subword3 = t[vowel_pos[1]+1:]
                    
                    if len(subword3) < 2:
                        t_chunk.append(subword1)                        
                        t_chunk.append(subword2 + subword3)
                    elif len(subword2) < 2:
                        t_chunk.append(subword1 + subword2)
                        t_chunk.append(subword3)
                    else:
                        t_chunk.append(subword1)
                        t_chunk.append(subword2)
                        t_chunk.append(subword3)
            #rules for words with >2 vowels
            else:
                if len(vowel_pos) > 2:
                    if vowel_pos[0] == 0:
                        subword1 = t[0:vowel_pos[1]+1]
                        t_chunk.append(subword1)
                        last_pos = vowel_pos[1]+1
                        for vow in vowel_pos[2:]:
                            subword_remain = t[last_pos:vow+1]
                            t_chunk.append(subword_remain)
                            last_pos = vow+1
                        subword_last = t[vowel_pos[-1]+1:]
                    else:
                        subword1 = t[0:vowel_pos[0]+1]
                        t_chunk.append(subword1)
                        last_pos = vowel_pos[0]+1
                        for vow in vowel_pos[1:]:
                            subword_remain = t[last_pos:vow+1]
                            t_chunk.append(subword_remain)
                            last_pos = vow+1                            
                        subword_last = t[vowel_pos[-1]+1:]
                        
                    #print "subword_last = ", subword_last
                    if len(subword_last) < 2:
                        t_chunk[len(t_chunk)-1] = t_chunk[-1] + subword_last
                    else:
                        t_chunk.append(subword_last)
    
    return t_chunk
    
def get_chunks_n_annotations(data_dir, processor_ver, remove_stopwords = False, replace_num = False, remove_non_alpha = False, to_lower = False, to_subwords = False):

    # Initialize the lists to be returned
    original_chunks=[]
    clean_chunks=[]
    clean_chunk_sents =[]
    temporal_states=[]
    event_files=[]    
    
    # Create a matrix with all the event news chunks with annotations
    if processor_ver == 1:
        events = process_annotated_files_dir(data_dir)
    elif processor_ver == 2:
        events = process_annotated_files_dir_v2(data_dir)
    else:
        events = process_annotated_files_dir_v3(data_dir)

    for i in xrange(len(events)):
        original_chunks.append(events[i][0])
        clean_chunks.append(process_chunk(events[i][0], remove_stopwords, replace_num, remove_non_alpha, to_lower, to_subwords))
        clean_chunk_sents += process_chunk_sent_level(events[i][0], remove_stopwords, replace_num, remove_non_alpha, to_lower, to_subwords, ret_word_list = True)         
        temporal_states.append(events[i][1])
        event_files.append(events[i][2])

    return (original_chunks, clean_chunks, clean_chunk_sents, temporal_states, event_files)

# Divide up the cleaned chunks into train and test sets        
def split_train_test_data(data_set, test_set_pct):
    
    train_set_size = int(np.ceil(len(data_set) * (1 - test_set_pct)))

    train_set = data_set[:train_set_size]
    test_set = data_set[train_set_size:]
    
    return (train_set, test_set)

#########################################################
# For CNN models
#########################################################

# Return a 2-D list where the binary columns represent the 
# annotations in the following order: 
# PA,OG,FP,FT,FM,NO,NA
def transform_annotations_to_binary(annotations):
        
    ret_list = []
    
    for i in annotations:
        if(i == "PA"):
            ret_list.append([1, 0, 0, 0, 0, 0, 0])
        elif(i == "OG"):
            ret_list.append([0, 1, 0, 0, 0, 0, 0])
        elif(i == "FP"):
            ret_list.append([0, 0, 1, 0, 0, 0, 0])
        elif(i == "FT"):
            ret_list.append([0, 0, 0, 1, 0, 0, 0])
        elif(i == "FM"):
            ret_list.append([0, 0, 0, 0, 1, 0, 0])
        elif(i == "NO"):
            ret_list.append([0, 0, 0, 0, 0, 1, 0])
        else:
            ret_list.append([0, 0, 0, 0, 0, 0, 1])
        
    return ret_list

def transform_digits_to_annotations(digits):
        
    ret_list = []
    
    for i in digits:
        if(i == 0):
            ret_list.append("PA")
        elif(i == 1):
            ret_list.append("OG")
        elif(i == 2):
            ret_list.append("FP")
        elif(i == 3):
            ret_list.append("FT")
        elif(i == 4):
            ret_list.append("FM")
        elif(i == 5):
            ret_list.append("NO")
        else:
            ret_list.append("NA")
        
    return ret_list

# Source: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]