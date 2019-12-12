import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math
from random import randint
#from decoder import *

SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 40
CLIP = 1.0
use_cuda = torch.cuda.is_available()


def train_nmt(input_variable,target_variable,input_lengths,model,criterion,optimizer,teacher_force_ratio=0.5):
    #specify this is the training stage
    model.train()
    #Zero gradients of the optimizer
    optimizer.zero_grad()
    #FeedForward
    loss= model(input_variable,input_lengths,target_variable,teacher_force_ratio,criterion=criterion)

    #Backpropagate the Loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
    optimizer.step()

    return loss.item()

################################Designed for New Imagine Model######################

def train_imagine_beam(input_variable, target_variable, im_variable, input_lengths,model,optimizer,criterion_mt,criterion_vse,loss_weight,teacher_force_ratio,max_length=MAX_LENGTH,clip=1):
    #Make model back to trianing 
    model.train()
    #Zero gradients of both optimizerse
    optimizer.zero_grad()
    
    #Get the output from the model
    loss,loss_mt,loss_vse = model(input_variable,input_lengths,target_variable,im_variable,teacher_force_ratio,criterion_mt=criterion_mt, criterion_vse=criterion_vse)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    #Optimize the Encoder and Decoder
    optimizer.step()
    
    return loss.data.item(),loss_mt.data.item(),loss_vse.data.item()

def train_imagine_beam_v2(input_variable, target_variable, im_variable, input_lengths,model,optimizer,criterion_mt,criterion_vse,teacher_force_ratio,max_length=MAX_LENGTH,clip=1,optimized_task="mt"):
    #Make model back to trianing 
    model.train()
    #Zero gradients of both optimizerse
    optimizer.zero_grad()

    #Get the output from the model
    loss,loss_mt,loss_vse = model(input_variable,input_lengths,target_variable,im_variable,teacher_force_ratio,criterion_mt=criterion_mt, criterion_vse=criterion_vse)
    if optimized_task == "mt":
        loss_mt.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),clip)

        #Optimize the Encoder and Decoder
        optimizer.step()
    else:
        loss_vse.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),clip)

        #Optimize the vse module, as well as Encoder and Decoder
        optimizer.step()


    return loss.data[0],loss_mt.data[0],loss_vse.data[0] 
################################################
#Randomly Display Some Results
def random_sample_display(test_data,output_list):
    sample_index = randint(0,len(test_data)-1)
    sample_source = test_data[sample_index][0]
    sample_ref = test_data[sample_index][1]
    sample_output_tokens = output_list[sample_index]
    sample_output = ' '.join(sample_output_tokens)
    return sample_source, sample_ref, sample_output
