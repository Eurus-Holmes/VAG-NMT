#Implements a New Structure for V9, where we will focus on Learning the Attention during Encoding and Let it Affect the Machine Translation
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
from ..layers import VSE_Imagine
from ..layers import VSE_Imagine_Enc
from ..utils.utils import l2norm

SOS_token = 2
EOS_token = 3
use_cuda = torch.cuda.is_available()

#Construct an Attention ImagineSeq2Seq Model
class NMT_AttentionImagine_Seq2Seq_Beam_V9(nn.Module):
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
                 dropout_rnn_enc=0.0, \
                 dropout_rnn_dec=0.0, \
                 dropout_im_emb = 0.0, \
                 dropout_txt_emb = 0.0, \
                 activation_vse = True, \
                 tied_emb=False):

        super(NMT_AttentionImagine_Seq2Seq_Beam_V9,self).__init__()
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
        self.tied_emb=tied_emb
        self.dropout_im_emb = dropout_im_emb
        self.dropout_txt_emb = dropout_txt_emb
        self.activation_vse = activation_vse
        self.attn_model = attn_model

        #Define all the parts. 
        self.encoder = LIUMCVC_Encoder(src_size,src_embedding_size,hidden_size,n_layers,dropout_rnn=dropout_rnn_enc, dropout_ctx=dropout_ctx, dropout_emb=dropout_emb)
        self.decoder = NMT_Decoder(tgt_size,tgt_embedding_size,hidden_size,2*hidden_size,n_layers,dropout_rnn=dropout_rnn_dec,dropout_out=dropout_out,dropout_emb=0.0,tied_emb=tied_emb)
        
        #Initialize the VSE_Imagine Module for encoder hidden states
        self.vse_imagine = VSE_Imagine_Enc(self.attn_model,self.im_feats_size,2*hidden_size,self.shared_embedding_size,self.dropout_im_emb,self.dropout_txt_emb,self.activation_vse)
        
        #Decoder Initialization Layer
        self.decoderini = nn.Linear(2*hidden_size,hidden_size)

        #Initilaize the layers with xavier method
        self.reset_parameters()
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name and param.data.dim() > 1:
                nn.init.kaiming_normal(param.data)
    
    def forward(self,src_var,src_lengths,tgt_var,im_var,teacher_force_ratio=1.0,max_length=80, criterion_mt=None, criterion_vse=None):
        '''
        Feed forward the input variable and compute the loss. tgt_var is always provided
        Input: 
            src_var: The minibatch input sentence indexes representation with size (B*W_s)
            src_lengths: The list of lenths of each sentence in the minimatch, the size is (B)
            im_var: The minibatch of the paired image ResNet Feature vecotrs, with the size(B*I), I is the image feature size.
            teacher_force_ratio: A scalar between 0 and 1 which defines the probability ot conduct the teacher_force traning.
            tgt_var: The output sentence groundtruth, if provided it will be used to help guide the training of the network. The Size is (B*W_t)
                     If not, it will just generate a target sentence which is shorter thatn max_length or stop when it finds a EOS_Tag.
            max_length: A integer value that specifies the longest sentence that can be generated from this network.     
        Output:  
            loss: Total loss which is the sum of loss_mt and loss_vse
            loss_mt: The loss for seq2seq machine translation
            loss_vse: The loss for visual-text embedding space learning          
        '''
        #Define the batch_size and input_length
        batch_size = src_var.size()[0]
        tgt_l = tgt_var.size()[1]
        loss = 0
        loss_mt = 0
        loss_vse = 0
        tgt_mask = (tgt_var != 0).float()

        #Update the self.tgt_l
        self.tgt_l = tgt_l


        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Conduct the VSE Step Here
        loss_vse,encoder_concat = self.vse_imagine(im_var,encoder_outputs,criterion_vse=criterion_vse,context_mask=context_mask)

        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        #decoder_hidden = F.tanh(self.decoderini(encoder_concat)).unsqueeze(0)
        
        #Initialize Decoder Hidden With Weighted Sum from the interaction with Images
        #decoder_hidden = F.tanh(self.decoderini(0.5*encoder_concat+0.5*encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()

        if tgt_var is not None:
            tgt_mask = (tgt_var != 0).float()

        """
        decoder_hiddens = Variable(torch.zeros(tgt_l,batch_size,self.hidden_size))
        if use_cuda:
            decoder_hiddens = decoder_hiddens.cuda()
        """

        #Determine whether teacher forcing is used. 
        is_teacher = random.random() < teacher_force_ratio
        if is_teacher: 
            decoder_hidden = F.tanh(self.decoderini(0.5*encoder_concat+0.5*encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, ctx_mask=context_mask)
                loss_n = criterion_mt(decoder_output,tgt_var[:,di])
                loss_mt += loss_n

                #decoder_hiddens[di] = decoder_hidden
                #text_embedding_sets[di] = text_embedding_di
                
                decoder_input = tgt_var[:,di]
        else:
            #Initialize Decoder Hidden With Mean States during the test cases
            decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, ctx_mask=context_mask)
                loss_n = criterion_mt(decoder_output,tgt_var[:,di])
                loss_mt += loss_n

                #Normalize The Text Embedding Vector
                #decoder_hiddens[di] = decoder_hidden
                #text_embedding_sets[di] = text_embedding_di

                _,top1 = decoder_output.data.topk(1)
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()
            
        #Average the machine translation loss
        #loss_mt = loss_mt / tgt_l
        loss_mt = (loss_mt / tgt_mask.sum(-1)).mean()

        loss = self.loss_w*loss_mt + (1-self.loss_w)*loss_vse

        return loss,loss_mt,loss_vse

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

    ###########################Function Defined Below is for Image Retrieval##############
    
    def embed_sent_im_eval(self,src_var,src_lengths,tgt_var,im_feats):
        """
            Embed the Target Sentences to the shared space
            Input: 
                source_sent: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_sent: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
            Output:
                txt_embedding.data: The embedded sentence tensor with size (B, SD), SD is the dimension of shared embedding
                space. 
                im_embedding.data: The embedded image tensor with size (B, SD), SD is the dimension of the shared embedding space
        """
        #Define the batch_size and input_length
        batch_size = src_var.size()[0]
        tgt_l = tgt_var.size()[1]

        #Update the self.tgt_l
        self.tgt_l = tgt_l


        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()


        decoder_hiddens = Variable(torch.zeros(tgt_l,batch_size,self.hidden_size))
        if use_cuda:
            decoder_hiddens = decoder_hiddens.cuda()
        
        #Determine whether teacher forcing is used. 
        for di in range(tgt_l):
            decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            #update text_embedding_sets
            decoder_hiddens[di] = decoder_hidden
            
            decoder_input = tgt_var[:,di]


        im_embedding,text_embedding = self.vse_imagine.get_emb_vec(im_feats, encoder_outputs,ctx_mask=context_mask)
        #im_embedding = l2norm(im_embedding)

        #I think may be another text_embedding here. 
        return im_embedding.data, text_embedding.data

    def embed_sent_im_test(self,src_var,src_lengths,im_feats,max_length=80):
        """
            Embed the Target Sentences to the shared space
            Input: 
                source_sent: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_sent: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
            Output:
                txt_embedding.data: The embedded sentence tensor with size (B, SD), SD is the dimension of shared embedding
                space. 
                im_embedding.data: The embedded image tensor with size (B, SD), SD is the dimension of the shared embedding space
        """
        #Define the batch_size and input_length
        batch_size = src_var.size()[0]
        tgt_l = max_length

        #Update the self.tgt_l
        self.tgt_l = tgt_l


        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()


        decoder_hiddens = Variable(torch.zeros(tgt_l,batch_size,self.hidden_size))
        if use_cuda:
            decoder_hiddens = decoder_hiddens.cuda()
        
        #Determine whether teacher forcing is used. 
        for di in range(tgt_l):
            decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_hiddens[di] = decoder_hidden
            #text_embedding_sets[di] = text_embedding_di
            _,top1 = decoder_output.data.topk(1)
            decoder_input = Variable(top1)
            if use_cuda:
                decoder_input = decoder_input.cuda()

        #Get the embedded vectors from vse_imagine
        im_embedding,text_embedding = self.vse_imagine.get_emb_vec(im_feats, encoder_outputs,ctx_mask=context_mask)
        
        #I think may be another text_embedding here. 
        return im_embedding.data, text_embedding.data
    ########################################################################
    def get_imagine_attention_eval(self,src_var,src_lengths,tgt_var,im_feats):
        """
            Get the attention_weights for validation dataset when tgt_var is available.
            Input: 
                source_var: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_var: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
            Output:
                output_translation: List of index for translations predicted by the seq2seq model
                attention_weights: (B,T)
        """
        #Define the batch_size and input_length
        batch_size = src_var.size()[0]
        tgt_l = tgt_var.size()[1]

        #Update the self.tgt_l
        self.tgt_l = tgt_l


        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()


        decoder_hiddens = Variable(torch.zeros(tgt_l,batch_size,self.hidden_size))
        if use_cuda:
            decoder_hiddens = decoder_hiddens.cuda()
        
        #Determine whether teacher forcing is used. 
        decoder_translation_list = []
        for di in range(tgt_l):
            decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            #update text_embedding_sets
            decoder_hiddens[di] = decoder_hidden
            #Update the transition list. 
            _,top1 = decoder_output.data.topk(1)
            decoder_translation_list.append(top1[:,0])
            #update the decoder_input for the next step
            decoder_input = tgt_var[:,di]

        #Get the attention weights
        attn_weights = self.vse_imagine.get_imagine_weights(im_feats,encoder_outputs,ctx_mask=context_mask)

        #Get the transition_list
        final_translations = []
        for b in range(batch_size):
            current_list = []
            for i in range(tgt_l):
                current_translation_token = decoder_translation_list[i][b]
                if current_translation_token == EOS_token:
                    break
                current_list.append(current_translation_token)
            final_translations.append(current_list)

        return attn_weights.data,final_translations

    def get_imagine_attention_test(self,src_var,src_lengths,im_feats,max_length=80):
        """
            Get the attention_weights for validation dataset when tgt_var is available.
            Input: 
                source_var: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_var: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
            Output:
                output_translation: List of index for translations predicted by the seq2seq model
                attention_weights: (B,T)
        """
        #Define the batch_size and input_length
        batch_size = src_var.size()[0]
        tgt_l = max_length

        #Update the self.tgt_l
        self.tgt_l = tgt_l


        #Encoder src_var
        encoder_outputs,context_mask = self.encoder(src_var,src_lengths)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()


        decoder_hiddens = Variable(torch.zeros(tgt_l,batch_size,self.hidden_size))
        if use_cuda:
            decoder_hiddens = decoder_hiddens.cuda()
        
        #Determine whether teacher forcing is used. 
        decoder_translation_list = []
        for di in range(tgt_l):
            decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            #update text_embedding_sets
            decoder_hiddens[di] = decoder_hidden
            #Update the transition list. 
            _,top1 = decoder_output.data.topk(1)
            decoder_translation_list.append(top1[:,0])
            #update the decoder_input for the next step
            decoder_input = Variable(top1)
            if use_cuda:
                decoder_input = decoder_input.cuda()

        #Get the attention weights
        attn_weights = self.vse_imagine.get_imagine_weights(im_feats,encoder_outputs,ctx_mask=context_mask)

        #Get the transition_list
        final_translations = []
        for b in range(batch_size):
            current_list = []
            for i in range(tgt_l):
                current_translation_token = decoder_translation_list[i][b]
                if current_translation_token == EOS_token:
                    break
                current_list.append(current_translation_token)
            final_translations.append(current_list)
            
        return attn_weights.data,final_translations