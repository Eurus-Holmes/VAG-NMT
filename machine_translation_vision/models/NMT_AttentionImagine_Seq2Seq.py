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

class ImagineAttn(nn.Module):
    def __init__(self,method,shared_embedding_size):
        super(ImagineAttn, self).__init__()
        self.method = method
        self.embedding_size = shared_embedding_size
        '''
        if self.method == 'dot':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        '''
    def forward(self,image_vec,decoder_hidden):
        '''
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(T,B,E)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,T) 
        '''

        #print("the hidden matrix size is: {}".format(hidden.size()))
        seq_len,batch_size = decoder_hidden.size()[0],decoder_hidden.size()[1] #Get the sequence length
        #Create variable to store attention energies
        attn_energies = self.score(image_vec.unsqueeze(2),decoder_hidden.transpose(0,1))
            
        #Normalize energies to weights in range 0 to 1, resize to B x 1 x T
        return F.softmax(attn_energies).unsqueeze(1)
    def score(self,image_vec,decoder_hidden):
        """
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,E,1), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(B,T,E)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,T) 
        """
        if self.method == 'dot':
            energy = decoder_hidden.bmm(image_vec)
            return energy.squeeze(2)


#Construct an Attention ImagineSeq2Seq Model
class NMT_AttentionImagine_Seq2Seq(nn.Module):
    def __init__(self, \
                 src_size, \
                 tgt_size, \
                 im_feats_size, \
                 src_embedding_size, \
                 tgt_embedding_size, \
                 hidden_size, \
                 shared_embedding_size, \
                 loss_w, \
                 beam_size=1, \
                 attn_model = 'dot', \
                 n_layers=1, \
                 dropout_ctx=0.0, \
                 dropout_emb=0.0, \
                 dropout_out=0.0, \
                 dropout_rnn=0.0):

        super(NMT_AttentionImagine_Seq2Seq,self).__init__()
        #Define all the parameters
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.im_feats_size = im_feats_size
        self.src_embedding_size = src_embedding_size
        self.tgt_embedding_size = tgt_embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.shared_embedding_size = shared_embedding_size
        self.beam_size = beam_size
        self.loss_w = loss_w

        #Define all the parts. 
        self.encoder = LIUMCVC_Encoder(src_size,src_embedding_size,hidden_size,n_layers,dropout_rnn=dropout_rnn, dropout_ctx=dropout_ctx, dropout_emb=dropout_emb)
        self.decoder = NMT_Decoder(tgt_size,tgt_embedding_size,hidden_size,2*hidden_size,n_layers,dropout_out=dropout_out, tied_emb=False)
        
        #Vision Embedding Layer
        self.im_embedding = nn.Linear(im_feats_size,shared_embedding_size)
        self.text_embedding = nn.Linear(hidden_size,shared_embedding_size)
        #Define the attention_mechanism
        self.imagine_attn = ImagineAttn(attn_model,shared_embedding_size)
        #Decoder Initialization Layer
        self.decoderini = nn.Linear(2*hidden_size,hidden_size)

    def forward(self,src_var,src_lengths,im_var=None,teacher_force_ratio=0,tgt_var=None,max_length=80, criterion_mt=None, criterion_vse=None):
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
        loss_mt = 0
        loss_vse = 0

        #Update the self.tgt_l
        self.tgt_l = tgt_l
        #Initialize the final_sample
        self.final_sample = []

        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()

        if tgt_var is not None:
            tgt_mask = (tgt_var != 0).float()

        if teacher_force_ratio > 0:
            text_embedding_sets = Variable(torch.zeros(tgt_l,batch_size,self.shared_embedding_size))
            if use_cuda:
                text_embedding_sets = text_embedding_sets.cuda()
            #Determine whether teacher forcing is used. 
            is_teacher = random.random() < teacher_force_ratio
            if is_teacher: 
                for di in range(tgt_l):
                    decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss_n = criterion_mt(decoder_output,tgt_var[:,di])
                    loss_mt += loss_n
                    #update text_embedding_sets
                    text_embedding_sets[di] = F.normalize(F.tanh(self.text_embedding(decoder_hidden.squeeze(0))))
                    decoder_input = tgt_var[:,di]
            else:
                for di in range(tgt_l):
                    decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss_n = criterion_mt(decoder_output,tgt_var[:,di])
                    loss_mt += loss_n
                    #update text_embedding_sets
                    text_embedding_sets[di] = F.normalize(F.tanh(self.text_embedding(decoder_hidden.squeeze(0))))
                    _,top1 = decoder_output.data.topk(1)
                    decoder_input = Variable(top1)
                    if use_cuda:
                        decoder_input = decoder_input.cuda()
            
            #Average the machine translation loss
            #loss_mt = loss_mt / tgt_l
            loss_mt = (loss_mt / tgt_mask.sum(-1)).mean()

            #Embed the Image vector to the shared space
            im_embedding = F.normalize(F.tanh(self.im_embedding(im_var))) #im_embedding: B*(2*hidden_size)
            #im_embedding = F.normalize(im_var)

            #Computed the attention weights
            attn_weights = self.imagine_attn(im_embedding,text_embedding_sets)
            #print(attn_weights)
            #Computed the weighted sum of hidden states as the final representation of the text vector
            text_embedding = attn_weights.bmm(text_embedding_sets.transpose(0,1)).squeeze(1)

            #Compute the Similarity Score.
            s_im_t = im_embedding.matmul(text_embedding.transpose(0,1))
            s_t_im = text_embedding.matmul(im_embedding.transpose(0,1))

            #Computet loss_vse
            #Compute the Visual Embedding Loss
            s_im_t_right = s_im_t.diag()
            s_im_t_right = s_im_t_right.repeat(s_im_t.size()[1],1).transpose(0,1)
            s_im_t_loss_M = s_im_t_right-s_im_t
            
            s_t_im_right = s_t_im.diag()
            s_t_im_right = s_t_im_right.repeat(s_t_im.size()[1],1).transpose(0,1)
            s_t_im_loss_M = s_t_im_right-s_t_im
            
            for i in range(s_im_t_loss_M.size()[0]):
                y = Variable(-1*torch.ones(s_im_t_loss_M.size()[1]))
                y[i] = 1
                if use_cuda:
                    y = y.cuda()
                loss_vse_i_1 = criterion_vse(s_im_t_loss_M[i],y)
                loss_vse_i_2 = criterion_vse(s_t_im_loss_M[i],y)
                loss_vse += loss_vse_i_1+loss_vse_i_2

            loss = self.loss_w*loss_mt + (1-self.loss_w)*loss_vse

        else:
            decoder_translation_list = []
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                #Compute the loss
                if tgt_var is not None:
                    loss_n = criterion_mt(decoder_output,tgt_var[:,di])
                    loss_mt += loss_n
                _,top1 = decoder_output.data.topk(1)
                #Append the current prediction to decoder_translation
                decoder_translation_list.append(top1[:,0])
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()

            #Compute the translation_prediction
            for b in range(batch_size):
                current_list = []
                for i in range(tgt_l):
                    current_translation_token = decoder_translation_list[i][b]
                    if current_translation_token == EOS_token:
                        break
                    current_list.append(current_translation_token)
                self.final_sample.append(current_list)
            #Only machine translation loss during the 
            #loss_mt = loss_mt/tgt_l
            if tgt_var is not None:
                loss_mt = (loss_mt / tgt_mask.sum(-1)).mean()

            loss = loss_mt
        return loss,loss_mt,loss_vse,self.final_sample

    def _validate_args(self,src_var,tgt_var,max_length):
        batch_size = src_var.size()[0]
        if tgt_var is None:
            tgt_l = max_length
        else:
            tgt_l = tgt_var.size()[1]

        return batch_size,tgt_l