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
UNK_token = 1
use_cuda = torch.cuda.is_available()

#Define a BeamSeq2Seq_2 Model
class NMT_Seq2Seq_Beam_V2(nn.Module):
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
                 dropout_rnn= 0.0, \
                 tied_emb = False):
        super(NMT_Seq2Seq_Beam_V2,self).__init__()
        #Define all the parameters
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.src_embedding_size = src_embedding_size
        self.tgt_embedding_size = tgt_embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.beam_size = beam_size
        self.tied_emb = tied_emb

        #Define all the parts. 
        self.encoder = LIUMCVC_Encoder(src_size,src_embedding_size,hidden_size,n_layers,dropout_rnn=dropout_rnn,dropout_ctx=dropout_ctx,dropout_emb=dropout_emb)
        self.decoder = NMT_Decoder(tgt_size,tgt_embedding_size,hidden_size,2*hidden_size,n_layers,dropout_out=dropout_out,tied_emb=tied_emb)
        #Decoder Initialization Layer
        self.decoderini = nn.Linear(2*hidden_size,hidden_size)

        #Initialize the weights
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name and param.data.dim() > 1:
                nn.init.kaiming_normal_(param.data)
    
    def forward(self,src_var,src_lengths,tgt_var, teacher_force_ratio=1.0,max_length=80,criterion=None):
        '''
        Feed forward the input variable and compute the loss. tgt_var is always provided. 
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
        batch_size = src_var.size()[0]
        tgt_l = tgt_var.size()[1]
        loss = 0
        tgt_mask = (tgt_var != 0).float()

        #Update the self.tgt_l
        self.tgt_l = tgt_l

        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)

        if use_cuda:
            decoder_input = decoder_input.cuda()

        #Determine whether teacher forcing is used. 
        is_teacher = random.random() < teacher_force_ratio
        if is_teacher: 
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs,ctx_mask=context_mask)
                #compute the curent loss
                loss_n = criterion(decoder_output,tgt_var[:,di])
                loss += loss_n
                decoder_input = tgt_var[:,di]
        else:
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs,ctx_mask=context_mask)
                #update the outputs
                loss_n = criterion(decoder_output,tgt_var[:,di])
                loss += loss_n
                _,top1 = decoder_output.data.topk(1)
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()
        
        #loss = loss/tgt_l
        loss = (loss / tgt_mask.sum(-1)).mean()

        return loss

    def _validate_args(self,src_var,tgt_var,max_length):
        batch_size = src_var.size()[0]
        if tgt_var is None:
            tgt_l = max_length
        else:
            tgt_l = tgt_var.size()[1]

        return batch_size,tgt_l

    def beamsearch_decode(self,src_var,src_lengths,beam_size=1,max_length=80,tgt_var=None):
        #Initiliaize the tgt_l
        tgt_l = max_length
        if tgt_var is not None:
            tgt_l = tgt_var.size()[1]
            
        batch_size = src_var.size()[0]

        self.tgt_l = tgt_l
        self.final_sample = []
        self.beam_size = beam_size

        #Encode the Sentences. 
        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)

        if use_cuda:
            decoder_input = decoder_input.cuda()

        if beam_size == 1:
            decoder_translation_list = []
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs,ctx_mask=context_mask)
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

        if beam_size > 1:
            self.final_sample = self.beamsearch(encoder_outputs,context_mask,decoder_input,decoder_hidden,beam_size,tgt_l)
        
        return self.final_sample

    def beamsearch(self,encoder_outputs,context_mask,decoder_input,decoder_hidden,beam_size,max_length,avoid_double=True,avoid_unk=False):
        #Define Batch_Size
        batch_size = encoder_outputs.size(1)
        n_vocab = self.tgt_size

        #Define Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = torch.arange(batch_size*beam_size).long() #[0:batch_size*beam_size]
        if use_cuda:
            nk_mask = nk_mask.cuda()
        pdxs_mask = (nk_mask/beam_size)*beam_size

        #Tile indices to use in the loop to expand first dim
        tile = nk_mask / beam_size

        #Define the beam
        beam = torch.zeros((max_length, batch_size, beam_size)).long()
        if use_cuda:
            beam = beam.cuda()

        #Create encoder outptus,context_mask with batch_dimension = batch_size*beam_size
        encoder_outputs_di = encoder_outputs[:,tile,:]
        context_mask_di = context_mask[:,tile] 

        #Define a inf numbers to assign to 
        inf = -1e5

        for di in range(max_length):
            if di == 0:
                decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs,ctx_mask=context_mask)
                nll,topk = decoder_output.data.topk(k=beam_size,sorted=False) #nll and topk have the shape [batch,topk]
                beam[0] = topk
            else:
                cur_tokens = beam[di-1].view(-1) #Get the input tokens to the next step
                fini_idxs = (cur_tokens == EOS_token).nonzero() #The index that checks whether the beam has terminated
                n_fini = fini_idxs.numel() #Verify if all the beams are terminated
                if n_fini == batch_size*beam_size:
                    break

                #Get the decoder fo the next iteration(batch_size*beam_size,1)
                decoder_input = Variable(cur_tokens,volatile=True)
                decoder_hidden = decoder_hidden[:,tile,:] #This operation will create a decoder_hidden states with size [batch_size*beam_size,H]
                
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden,encoder_outputs_di, ctx_mask=context_mask_di)
                decoder_output = decoder_output.data

                #Suppress probabilities of previous tokens at current time step, which avoids generating repeated word. 
                if avoid_double:
                    decoder_output.view(-1).index_fill_(0,cur_tokens+(nk_mask*n_vocab),inf)

                #Suppress probabilities of unk word.
                if avoid_unk:
                    decoder_output[:,UNK_token] = inf

                """
                Favor finished hyps to generate <eos> again
                Their nll scores will not increase further and they will always be kept in the beam.
                This operation assures the future generation for those finished hypes will always pick EOS_token. 
                """
                if n_fini > 0:
                    fidxs = fini_idxs[:,0]
                    decoder_output.index_fill_(0,fidxs,inf)
                    decoder_output.view(-1).index_fill_(0,fidxs*self.tgt_size+EOS_token,0)

                #Update the current score
                nll = (nll.unsqueeze(2) + decoder_output.view(batch_size,-1,n_vocab)).view(batch_size,-1) #Size is [batch,beam*n_vocab]

                #Pick the top beam_size best scores
                nll,idxs = nll.topk(beam_size,sorted=False) #nll, idxs have the size [batch_size,beam_size]

                #previous indices into the beam and current token indices
                pdxs = idxs / n_vocab #size is [batch_size,beam_size]

                #Update the previous token in beam[di]
                beam[di] = idxs % n_vocab

                # Permute all hypothesis history according to new order
                beam[:di] = beam[:di].gather(2,pdxs.repeat(di,1,1))

                # Compute correct previous indices
                #Mask is nedded since we are in flatten regime
                tile = pdxs.view(-1) + pdxs_mask
        #Put an explicit <eos> to ensure that every sentence end in the end
        beam[max_length-1] = EOS_token

        #Find lengths by summing tokens not in (pad,bos,eos)
        lens = (beam.transpose(0,2) > 3).sum(-1).t().float().clamp(min=1)

        #Normalize Scores by length
        nll /= lens.float()
        top_hyps = nll.topk(1, sorted=False)[1].squeeze(1)
        #Get best hyp for each sample in the batch
        hyps = beam[:,range(batch_size),top_hyps].cpu().numpy().T

        final_sample = []
        
        for b in range(batch_size):
            current_list = []
            for i in range(max_length):
                current_translation_token = hyps[b][i]
                if current_translation_token == EOS_token:
                    break
                current_list.append(current_translation_token)
            final_sample.append(current_list)

        return final_sample