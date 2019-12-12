#This NMT Decoder will return two things: (1) The second decoder hidden state (2) The first decoder hidden state

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

class BahdanauAttn(nn.Module):
    def __init__(self, context_size, hidden_size):
        super(BahdanauAttn, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size

        #attn_h,attn_e and v are the three parameters to tune. 
        self.attn_h = nn.Linear(self.hidden_size, self.context_size,bias=False)
        self.attn_e = nn.Linear(self.context_size,self.context_size,bias=False)
        self.v = nn.Parameter(torch.rand(self.context_size))
        #Normalize Data
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        

        # end of update
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs,ctx_mask=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (1,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (S,B,C)
        :return
            attention energies in shape (B,S)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        if ctx_mask is not None:
            self.mask = (1-ctx_mask.transpose(0,1).data).byte()
            attn_energies.data.masked_fill_(self.mask,-float('inf'))
        return self.softmax(attn_energies).unsqueeze(1) # normalize with softmax, attn_energies = B * 1 * T

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn_h(hidden)+self.attn_e(encoder_outputs)) #The size of energy is B*T*C
        energy = energy.transpose(2,1) # [B*C*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*C]
        energy = torch.bmm(v,energy) # [B*1*S]
        return energy.squeeze(1) #[B*S]

#Implement of cGRU from nematus language toolkit paper address is: 
class NMT_Decoder_V2(nn.Module):
    def __init__(self,output_size, \
                 embedding_size, \
                 hidden_size, \
                 context_size, \
                 n_layers = 1, \
                 dropout_emb = 0.0, \
                 dropout_rnn = 0.0, \
                 dropout_out = 0.0, \
                 bias_zero = True, \
                 tied_emb = False):
                 #dropout_emb is 0.0 as defaul value   
        super(NMT_Decoder_V2,self).__init__()

        #Keep Parameters for Reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size 
        self.context_size = context_size 
        self.n_layers=n_layers
        self.dropout_emb = dropout_emb
        self.dropout_out=dropout_out
        self.bias_zero = bias_zero
        self.tied_emb = tied_emb
        #Define layers
        self.embedding=nn.Embedding(output_size,embedding_size,padding_idx=0)
        if self.dropout_emb > 0.0:
            self.embedding_dropout = nn.Dropout(dropout_emb)

        #The Three main components for the cGRU.
        self.gru_1 = nn.GRU(embedding_size,hidden_size,num_layers=n_layers,dropout=dropout_rnn)
        self.attn = BahdanauAttn(context_size, hidden_size)
        self.context2hid = nn.Linear(context_size,hidden_size, bias=False)
        self.gru_2 = nn.GRU(hidden_size,hidden_size,num_layers=n_layers,dropout=dropout_rnn)
        
        #Three matrix to generate a intermediate representation tj for final output
        self.W1 = nn.Linear(hidden_size,embedding_size)
        if self.bias_zero:
            torch.nn.init.constant(self.W1.bias.data, 0.0)
        self.W2 = nn.Linear(context_size,embedding_size)
        if self.bias_zero:
            torch.nn.init.constant(self.W2.bias.data, 0.0)
        self.W3 = nn.Linear(embedding_size,embedding_size)
        if self.bias_zero:
            torch.nn.init.constant(self.W3.bias.data, 0.0)

        #Output Layer
        self.out = nn.Linear(embedding_size,output_size)
        if self.bias_zero:
            torch.nn.init.constant(self.out.bias.data, 0.0)
        if self.dropout_out > 0.0:
            self.output_dropout = nn.Dropout(dropout_out)
        if self.tied_emb:
            self.out.weight = self.embedding.weight
        #Output Dropout Layer, Inspired from LIUMVIC
        #self.out_drop = nn.Dropout(0.5)
    def forward(self,word_input,last_hidden,encoder_outputs,ctx_mask=None):
        '''
        Input:
            word_input: A tensor with size B*1, representing the previous predicted word 
            last_hidden: The hidden state vector from the previous timestep, s_t_1
            encoder_outputs: Size T_in*B*Context_Size
        '''
        batch_size = word_input.size()[0]
        #Embedding Word input to WordVectors
        word_embedding = self.embedding(word_input).view(1,batch_size,-1)

        #Process the word_embedding through the first gru to generate the intermediate representation
        gru_1_output,gru_1_hidden = self.gru_1(word_embedding,last_hidden) #The gru_1_hidden is the intermediate hidden state, with size(L,B,N), N is the hidden_size, L is the layer_size
        
        #Compute the Attentional Weights Matrix
        attn_weights = self.attn(gru_1_hidden,encoder_outputs,ctx_mask=ctx_mask)
        #Get the update context
        context = attn_weights.bmm(encoder_outputs.transpose(0,1)) # context size B*1*C
        context_hidden = self.context2hid(context) # Convert the context to hidden size
        #Compute the output from second gru. 
        gru_2_output,gru_2_hidden = self.gru_2(context_hidden.transpose(0,1),gru_1_hidden)

        #Squeeze the Size
        gru_2_output = gru_2_output.squeeze(0) #1*B*H -> B*H
        context = context.squeeze(1) #B*1*C -> B*C
        word_embedding = word_embedding.squeeze(0) #1*B*E -> B*E
        
        #Compute the intermediate representation before softmax
        concat_output = F.tanh(self.W1(gru_2_output)+self.W3(word_embedding)+self.W2(context))
        #concat_output = F.tanh(self.W1(gru_2_output)) #Adjust this once more with respect to OZAN's implementation.

        if self.dropout_out > 0.0:
            concat_output=self.output_dropout(concat_output)

        output = F.log_softmax(self.out(concat_output),dim=-1)

        #Return a Stacked hidden states. 
        gru_stacked_hidden = torch.cat((gru_1_hidden,gru_2_hidden),dim=-1)

        return output,gru_2_hidden,gru_stacked_hidden