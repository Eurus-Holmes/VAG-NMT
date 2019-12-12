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
class NMT_AttentionImagine_Seq2Seq_Beam(nn.Module):
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
                 tied_emb=False):

        super(NMT_AttentionImagine_Seq2Seq_Beam,self).__init__()
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
        self.tied_emb=False
        self.dropout_im_emb = dropout_im_emb
        self.dropout_txt_emb = dropout_txt_emb

        #Define all the parts. 
        self.encoder = LIUMCVC_Encoder(src_size,src_embedding_size,hidden_size,n_layers,dropout_rnn=dropout_rnn_enc, dropout_ctx=dropout_ctx, dropout_emb=dropout_emb)
        self.decoder = NMT_Decoder(tgt_size,tgt_embedding_size,hidden_size,2*hidden_size,n_layers,dropout_rnn=dropout_rnn_dec,dropout_out=dropout_out,dropout_emb=0.0,tied_emb=tied_emb)
        
        #Vision Embedding Layer
        self.im_embedding = nn.Linear(im_feats_size,shared_embedding_size)
        #Include a dropout Layer 
        if self.dropout_im_emb > 0:
            self.im_embedding_dropout = nn.Dropout(self.dropout_im_emb)

        self.text_embedding = nn.Linear(hidden_size,shared_embedding_size)
        if self.dropout_txt_emb > 0:
            self.txt_embedding_dropout = nn.Dropout(self.dropout_txt_emb)

        #Define the attention_mechanism
        self.imagine_attn = ImagineAttn(attn_model,shared_embedding_size)
        #Decoder Initialization Layer
        self.decoderini = nn.Linear(2*hidden_size,hidden_size)

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
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        decoder_hidden = F.tanh(self.decoderini(encoder_outputs.sum(0)/context_mask.sum(0).unsqueeze(1))).unsqueeze(0)
        
        #Initialize the output
        if use_cuda:
            decoder_input = decoder_input.cuda()

        if tgt_var is not None:
            tgt_mask = (tgt_var != 0).float()


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
                text_embedding_di = F.tanh(self.text_embedding(decoder_hidden.squeeze(0)))
                if self.dropout_txt_emb > 0:
                    text_embedding_di = self.txt_embedding_dropout(text_embedding_di)

                #Normalize The Text Embedding Vector
                text_embedding_sets[di] = F.normalize(text_embedding_di)
                
                decoder_input = tgt_var[:,di]
        else:
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss_n = criterion_mt(decoder_output,tgt_var[:,di])
                loss_mt += loss_n
                #update text_embedding_sets
                text_embedding_di = F.tanh(self.text_embedding(decoder_hidden.squeeze(0)))
                if self.dropout_txt_emb > 0:
                    text_embedding_di = self.txt_embedding_dropout(text_embedding_di)

                #Normalize The Text Embedding Vector
                text_embedding_sets[di] = F.normalize(text_embedding_di)

                _,top1 = decoder_output.data.topk(1)
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()
            
        #Average the machine translation loss
        #loss_mt = loss_mt / tgt_l
        loss_mt = (loss_mt / tgt_mask.sum(-1)).mean()

        #Embed the Image vector to the shared space
        im_embedding = F.tanh(self.im_embedding(im_var))
        if self.dropout_im_emb > 0:
            im_embedding = self.im_embedding_dropout(im_embedding)
        #Normalize the image embedding vector
        im_embedding = F.normalize(im_embedding) #im_embedding: B*(2*hidden_size)
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
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
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
        else:
            for b in range(batch_size):
                decoder_input_b = decoder_input[b]
                decoder_hidden_b = decoder_hidden[:,b,:].unsqueeze(1)
                encoder_outputs_b = encoder_outputs[:,b,:].unsqueeze(1)
                output_b,final_sample_b = self.beamsearch(decoder_input_b,decoder_hidden_b,encoder_outputs_b)
                self.final_sample.append(final_sample_b)


        return self.final_sample

    def _inflate(self,tensor, times, dim):
        """
        Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
        Args:
            tensor: A :class:`Tensor` to inflate
            times: number of repetitions
            dim: axis for inflation (default=0)
        Returns:
            A :class:`Tensor`
        Examples::
            >> a = torch.LongTensor([[1, 2], [3, 4]])
            >> a
            1   2
            3   4
            [torch.LongTensor of size 2x2]
            >> b = ._inflate(a, 2, dim=1)
            >> b
            1   2   1   2
            3   4   3   4
            [torch.LongTensor of size 2x4]
            >> c = _inflate(a, 2, dim=0)
            >> c
            1   2
            3   4
            1   2
            3   4
            [torch.LongTensor of size 4x2]
        """
        repeat_dims = [1] * tensor.dim()
        repeat_dims[dim] = times
        return tensor.repeat(*repeat_dims)

    def beamsearch(self,decoder_input,decoder_hidden,encoder_outputs):
        """
        Conduct beamsearch decoding process for each instance. 
        Input:
            decoder_input: beginning input words, which should be SOS_token 
            decoder_hidden: beginning hidden state for one instance, which has size:(1,1,hidden_size)
            encoder_outputs: Variable of encoder_outputs , (src_l,1,hidden_size)
        Output:
            output_b: the finalized output for instance b, with size(tgt_l,1,output_size)
            final_sample: the final decoded sequence for current instance b, which is a list of output index
        """
        #Initialize Parameters for Beam_Search Decoder
        self.dead_k = 0 #Define when each sampling exists

        #Initialize the beam object where we need to track 
        self.final_score_b = list()
        self.sample_b = list()
        self.sample_output_b = list()
        
        #Initialize a set of list for store intermediate results
        score_candidate_b = list()
        score_candidate_var = None
        hyp_sample_b = list()
        hyp_output_b = list()

        #Initialize the decoder_input, decoder_hidden and encoder_outputs for beam_size k
        decoder_input_k = self._inflate(decoder_input,self.beam_size,0)
        decoder_hidden_k = self._inflate(decoder_hidden,self.beam_size,1)
        encoder_outputs_k = self._inflate(encoder_outputs,self.beam_size,1)

        #Verify the input
        di = 0
        while di < self.tgt_l and self.dead_k < self.beam_size:
            decoder_output,decoder_hidden = self.decoder(decoder_input_k,decoder_hidden_k,encoder_outputs_k)
            #initialize the next terms
            next_word = []
            next_hidden = []
            #Initialize the new candidates
            new_score_candidate_b = []
            new_hyp_output_b = []
            new_hyp_sample_b = []

            if di==0:
                #Find the TopK Index
                topk_prob,topk_index = decoder_output.data[0].topk(self.beam_size-self.dead_k)
                for i in range(self.beam_size-self.dead_k):
                    #Append the score to new_score_candidate_b
                    new_score_candidate_b.append(topk_prob[i])
                    #Append the corresponding output to new_hyp_output_b
                    new_hyp_output_b.append([decoder_output[0].unsqueeze(0)])   
                    #Append the corresponding candidate to new_hyp_sample_b
                    new_hyp_sample_b.append([topk_index[i]])
                    #Check if this beam ends
                    if topk_index[i] == EOS_token:
                        #Update Sample_b
                        self.sample_b.append(new_hyp_sample_b[i])    
                        #Update Sample_Output_b
                        self.sample_output_b.append(new_hyp_output_b[i])
                        #Update the final score
                        self.final_score_b.append(new_score_candidate_b[i])
                        #update dead_k
                        self.dead_k += 1
                    else:
                        hyp_sample_b.append(new_hyp_sample_b[i])
                        hyp_output_b.append(new_hyp_output_b[i])
                        score_candidate_b.append(new_score_candidate_b[i])
                        #Update next_word and next_hidden
                        next_word.append(new_hyp_sample_b[i][-1])
                        next_hidden.append(decoder_hidden[:,0,:].unsqueeze(1))
                
                #Create new decoder_input_k and decoder_hidden_k
                decoder_input_k = Variable(torch.LongTensor(next_word))
                #Create a variable to represent the prevscores
                score_candidate_var = self._inflate(Variable(torch.FloatTensor(score_candidate_b)).unsqueeze(1),self.tgt_size,1)
                if use_cuda:
                    decoder_input_k = decoder_input_k.cuda()
                    score_candidate_var = score_candidate_var.cuda()

                decoder_hidden_k = torch.cat(next_hidden,dim=1)
                #update the encoder_outputs_k
                encoder_outputs_k = self._inflate(encoder_outputs,self.beam_size-self.dead_k,1)

            else:
                #Conver the hyper to a prevous state
                pre_score_candidate_b = score_candidate_b
                pre_hyper_sample_b = hyp_sample_b
                pre_hyper_output_b = hyp_output_b

                #Empty score_candidate_b,hyper_sample_b,hyper_output_b
                hyp_sample_b = []
                hyp_output_b = []
                score_candidate_b = []

                #compute the score
                decoder_output_score = score_candidate_var+decoder_output
                flatten_decoder_output_score = decoder_output_score.view(1,-1)
                
                #print(flatten_decoder_output)
                topk_prob,topk_index = flatten_decoder_output_score.data[0].topk(self.beam_size-self.dead_k)
                for i in range(self.beam_size-self.dead_k):
                    #retrieval the back_pointer
                    current_word = topk_index[i]%self.tgt_size
                    current_back_pointer = int(topk_index[i]/self.tgt_size)
                    #Update the new_hyp series
                    new_score_candidate_b.append(topk_prob[i])
                    
                    current_sample_i = list(pre_hyper_sample_b[current_back_pointer])
                    current_sample_i.append(current_word)
                    new_hyp_sample_b.append(current_sample_i)

                    current_output_i = list(pre_hyper_output_b[current_back_pointer])
                    current_output_i.append(decoder_output[current_back_pointer].unsqueeze(0))
                    new_hyp_output_b.append(current_output_i)
                    
                    if current_word == EOS_token:
                        #Update Sample_b
                        self.sample_b.append(new_hyp_sample_b[i])    
                        #Update Sample_Output_b
                        self.sample_output_b.append(new_hyp_output_b[i])
                        #Update the final score
                        self.final_score_b.append(new_score_candidate_b[i])
                        #update dead_k
                        self.dead_k += 1
                    else:
                        hyp_sample_b.append(new_hyp_sample_b[i])
                        hyp_output_b.append(new_hyp_output_b[i])
                        score_candidate_b.append(new_score_candidate_b[i])
                        #Update next_word and next_hidden
                        next_word.append(new_hyp_sample_b[i][-1])
                        next_hidden.append(decoder_hidden[:,0,:].unsqueeze(1))
                
                #Create a variable to represent the prevscores
                if len(score_candidate_b) > 0:
                    #Create new decoder_input_k and decoder_hidden_k
                    decoder_input_k = Variable(torch.LongTensor(next_word))
                    score_candidate_var = self._inflate(Variable(torch.FloatTensor(score_candidate_b)).unsqueeze(1),self.tgt_size,1)
                    
                    if use_cuda:
                        decoder_input_k = decoder_input_k.cuda()
                        score_candidate_var = score_candidate_var.cuda()

                    decoder_hidden_k = torch.cat(next_hidden,dim=1)
                    #update the encoder_outputs_k
                    encoder_outputs_k = self._inflate(encoder_outputs,self.beam_size-self.dead_k,1)
            #update the time step
            di += 1
            '''
            print("current processing samples: {}".format(hyp_sample_b))
            print("current processing scores: {}".format(score_candidate_b))
            print("selected final samples: {}".format(self.sample_b))
            print("selected final samples score: {}".format(self.final_score_b))
            '''

        #Update the sample_b, final_score_b, and sample_output_b if early stop does not happen
        if self.dead_k < self.beam_size:
            #update sample_b
            self.sample_b += hyp_sample_b
            #update final_scores
            self.final_score_b += score_candidate_b
            #update sample_output
            self.sample_output_b += hyp_output_b
        '''
        print("selected final samples: {}".format(self.sample_b))
        print("selected final samples score: {}".format(self.final_score_b))
        print("final_output_length: {}".format(len(self.sample_output_b)))
        '''
        #Pick the best candidate out of the sample
        normalized_score = []
        for i,x in enumerate(self.final_score_b):
            normalized_score.append(x/len(self.sample_b[i]))
        #print(normalized_score)
        #Find the max_index for the normalized_score
        max_index = normalized_score.index(max(normalized_score))
        #print(max_index)

        #Construct the output_b and final_sample_b
        final_sample_b = self.sample_b[max_index]
        output_b = torch.cat(self.sample_output_b[max_index],dim=0).unsqueeze(1)
        #print(output_b.size())
        #When return final_sample_b, get rid of the last temr, as it is just a EOS_token
        return output_b,final_sample_b[:-1]