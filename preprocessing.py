"""
This program file contains all the functions implemented to load and preprocess the dataset for machine_translation_vision_project
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
from nltk.tokenize import sent_tokenize
import re
from machine_translation_vision.samplers import BucketBatchSampler


#MAX_LENGTH = 50
#Define a couple of parameters
SOS_token = 2
EOS_token = 3
UNK_token = 1
PAD_Token = 0
use_cuda = torch.cuda.is_available()

#Load the dataset in a text file located with data_path
def load_data(data_path):
    with open(data_path,'r') as f:
        data = f.readlines()
    return data
def format_data(data_x,data_y,IKEA=False):
    if not IKEA:
        data=[[x.strip(),y.strip()] for x,y in zip(data_x,data_y)]
    else:
        data=[]
        for x,y in zip(data_x,data_y):
            #conver the paragraph into sentences
            x_s = sent_tokenize(x)
            y_s = sent_tokenize(y)
            #Check if len of the list is the same
            if len(x_s) == len(y_s):
                data += [[x.strip(),y.strip()] for x,y in zip(x_s,y_s)]
    return data
#Construct Word2Id and Id2Word Dictionaries from a loaded vocab file
def construct_vocab_dic(vocab):
    word2id = {}
    id2word = {}
    for i,word in enumerate(vocab):
        word2id[word.strip()] = i+1
        id2word[i+1] = word.strip()
    return word2id,id2word

#Filter out the pairs that has a sentence having
def data_filter(data,max_length):
    new_data = []
    for d in data:
        if len(d[0].split()) <= max_length and len(d[1].split()) <= max_length:
            new_data.append(d)
    return new_data

def indexes_from_sentence(vocab, sentence):
    return [vocab.get(word,UNK_token) for word in sentence.split(' ')]

def variable_from_sentence(vocab, sentence):
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if use_cuda: var = var.cuda()
    return var

def variables_from_pair(pair,s_vocab, t_vocab):
    input_variable = variable_from_sentence(s_vocab, pair[0])
    target_variable = variable_from_sentence(t_vocab, pair[1])
    return (input_variable, target_variable)

#Create data pairs with each pair represented by corresponding wordids in each language. 
def create_data_index(pairs,source_vocab,target_vocab):
    source_indexes = [indexes_from_sentence(source_vocab,x[0])+[EOS_token] for x in pairs]
    target_indexes = [indexes_from_sentence(target_vocab,x[1])+[EOS_token] for x in pairs]
    return [[s,t] for s,t in zip(source_indexes,target_indexes)]

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq_new = seq+[0 for i in range(max_length - len(seq))]
    return seq_new

def data_generator(data_pairs, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size/batch_size)
    for i in range(0,data_size,batch_size):
        if i+batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i+batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i+batch_size]]
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              


        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 


        #Generate batch_x and batch_y
        batch_x,batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)),Variable(torch.LongTensor(batch_y_pad_sorted))
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        #Yield the batch data|
        yield batch_x,batch_y,list(reversed(sorted(batch_x_lengths))),batch_y_lengths_sorted,x_reverse_sorted_index

def data_generator_tl(data_pairs,batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length.  
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              


        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 


        #Generate batch_x and batch_y
        batch_x,batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)),Variable(torch.LongTensor(batch_y_pad_sorted))
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        #Yield the batch data|
        yield batch_x,batch_y,list(reversed(sorted(batch_x_lengths))),batch_y_lengths_sorted

def data_generator_single(batch_data_x):
    x_length = max([len(x) for x in batch_data_x])

    #Get a list of tokens
    batch_x_pad = []
    batch_x_lengths = []
    
    #Updated batch_x_lengths, batch_x_pad
    for x_tokens in batch_data_x:
        x_l = len(x_tokens)
        x_pad_seq = pad_seq(x_tokens,x_length)
        batch_x_lengths.append(x_l)
        batch_x_pad.append(x_pad_seq)
    #Reorder the lengths
    x_sorted_index = list(np.argsort(batch_x_lengths))
    x_reverse_sorted_index = list(reversed(x_sorted_index))
    batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

    batch_x = Variable(torch.LongTensor(batch_x_pad_sorted))
    if use_cuda:
        batch_x = batch_x.cuda()
    return batch_x,list(reversed(sorted(batch_x_lengths))),x_reverse_sorted_index

def data_generator_mtv(data_pairs, data_im, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size/batch_size)
    for i in range(0,data_size,batch_size):
        if i+batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i+batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i+batch_size]]
            batch_data_im = torch.from_numpy(data_im[i:i+batch_size])
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]
            batch_data_im = torch.from_numpy(data_im[i:data_size])
            
        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
        
        #Generate batch_x and batch_y
        batch_x,batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)),Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
        
        #Yield the batch data|
        yield batch_x,batch_y,batch_im,list(reversed(sorted(batch_x_lengths))),batch_y_lengths_sorted,x_reverse_sorted_index

def data_generator_tl_mtv(data_pairs, data_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
        
        #Generate batch_x and batch_y
        batch_x,batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)),Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
        
        #Yield the batch data|
        yield batch_x,batch_y,batch_im,list(reversed(sorted(batch_x_lengths))),batch_y_lengths_sorted

def data_generator_tl_mtv_imretrieval(data_pairs, data_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        #print(bidx)
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              
        #print(x_reverse_sorted_index)

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
        
        #Generate batch_x and batch_y
        batch_x,batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)),Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
        index_retrieval = [bidx[x] for x in x_reverse_sorted_index]
        #print(index_retrieval)

        #Yield the batch data|
        yield batch_x,batch_y,batch_im,list(reversed(sorted(batch_x_lengths))),index_retrieval

def translation_reorder(translation,length_sorted_index,id2word):
    #Reorder translation
    original_translation = [None]*len(translation)
    for i,t in zip(length_sorted_index,translation):
        original_translation[i] = [id2word.get(x,'<unk>') for x in t]
    return original_translation

def translation_reorder_BPE(translation,length_sorted_index,id2word):
    #Reorder translation
    original_translation = [None]*len(translation)
    for i,t in zip(length_sorted_index,translation):
        BPE_translation_tokens = [id2word.get(x,'<unk>') for x in t]
        #Processing the original translation such that
        BPE_translation = ' '.join(BPE_translation_tokens)
        #Search and Replace patterns
        ori_translation = re.sub(r'@@ ',"",BPE_translation)
        #Tokenlize the ori_translation and keep it in the orginal_translation list
        original_translation[i] = ori_translation.split()

    return original_translation