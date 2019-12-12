import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math
import random
import sys

from ..layers import LIUMCVC_Encoder
from ..layers import NMT_Decoder

SOS_token = 2
EOS_token = 3
use_cuda = torch.cuda.is_available()

#Define a BeamSeq2Seq_2 Model
class NMT_Seq2Seq(nn.Module):
    def __init__(self, \
                 src_size, \
                 tgt_size, \
                 src_embedding_size, \
                 tgt_embedding_size, \
                 hidden_size, \
                 beam_size=1, \
                 n_layers=1, \
                 dropout_ctx= 0.0, \
                 dropout_emb= 0.0, \
                 dropout_out= 0.0, \
                 dropout_rnn= 0.0):
        super(NMT_Seq2Seq,self).__init__()
        #Define all the parameters
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.src_embedding_size = src_embedding_size
        self.tgt_embedding_size = tgt_embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.beam_size = beam_size

        #Define all the parts. 
        self.encoder = LIUMCVC_Encoder(src_size,src_embedding_size,hidden_size,n_layers,dropout_rnn=dropout_rnn,dropout_ctx=dropout_ctx,dropout_emb=dropout_emb)
        self.decoder = NMT_Decoder(tgt_size,tgt_embedding_size,hidden_size,2*hidden_size,n_layers,dropout_out=dropout_out)
        #Decoder Initialization Layer
        self.decoderini = nn.Linear(2*hidden_size,hidden_size)

    def forward(self,src_var,src_lengths,teacher_force_ratio=0,tgt_var=None,max_length=80,criterion=None):
        '''
        Input: 
            src_var: The minibatch input sentence indexes representation with size (B*W_s)
            src_lengths: The list of lenths of each sentence in the minimatch, the size is (B)
            im_var: The minibatch of the paired image ResNet Feature vecotrs, with the size(B*I), I is the image feature size.
            teacher_force_ratio: A scalar between 0 and 1 which defines the probability ot conduct the teacher_force traning.
            tgt_var: The output sentence groundtruth, if provided it will be used to help guide the training of the network. The Size is (B*W_t)
                     If not, it will just generate a target sentence which is shorter thatn max_length or stop when it finds a EOS_Tag.
            max_length: A integer value that specifies the longest sentence that can be generated from this network.     
        Output:            
        '''
        #Define the batch_size and input_length
        batch_size,tgt_l = self._validate_args(src_var,tgt_var,max_length)
        loss = 0

        #Update the self.tgt_l
        self.tgt_l = tgt_l
        #Initialize the final_sample
        self.final_sample = []

        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)

        if use_cuda:
            decoder_input = decoder_input.cuda()
        '''
        if tgt_var is not None:
            tgt_mask = (tgt_var != 0).float()
        '''
        if teacher_force_ratio > 0:
            #Determine whether teacher forcing is used. 
            is_teacher = random.random() < teacher_force_ratio
            if is_teacher: 
                for di in range(tgt_l):
                    decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    #compute the curent loss
                    loss_n = criterion(decoder_output,tgt_var[:,di])
                    loss += loss_n
                    decoder_input = tgt_var[:,di]
            else:
                for di in range(tgt_l):
                    decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    #update the outputs
                    loss_n = criterion(decoder_output,tgt_var[:,di])
                    loss += loss_n
                    _,top1 = decoder_output.data.topk(1)
                    decoder_input = Variable(top1)
                    if use_cuda:
                        decoder_input = decoder_input.cuda()
            #loss = (loss / tgt_mask.sum(-1)).mean()
            loss = loss/tgt_l
        
        else:
            """
            Implement Beam Search Decoder. The Decoder will continue work untill all the batches's candidates reach the EOS Tag
            """
            #Conduct the beam search process for each batch separately. 
            decoder_translation_list = []
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                #Compute the loss
                if tgt_var is not None:
                    loss_n = criterion(decoder_output,tgt_var[:,di])
                    loss += loss_n
                _,top1 = decoder_output.data.topk(1)
                #Append the current prediction to decoder_translation
                decoder_translation_list.append(top1[:,0])
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()
            if tgt_var is not None:
                #loss = (loss / tgt_mask.sum(-1)).mean()
                loss = loss / tgt_l
            
            #Compute the translation_prediction
            for b in range(batch_size):
                current_list = []
                for i in range(tgt_l):
                    current_translation_token = decoder_translation_list[i][b]
                    if current_translation_token == EOS_token:
                        break
                    current_list.append(current_translation_token)
                self.final_sample.append(current_list)

        return loss,self.final_sample

    def _validate_args(self,src_var,tgt_var,max_length):
        batch_size = src_var.size()[0]
        if tgt_var is None:
            tgt_l = max_length
        else:
            tgt_l = tgt_var.size()[1]

        return batch_size,tgt_l