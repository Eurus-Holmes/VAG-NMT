##Implement a VSE learning module using the mean of the decoder hidden states##

import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import l2norm

class VSE_Imagine_Mean(nn.Module):
    def __init__(self,attn_type,\
                im_size,\
                hidden_size,\
                shared_embedding_size,\
                dropout_im_emb=0.0,\
                dropout_txt_emb=0.0, \
                activation_vse = True):
        super(VSE_Imagine_Mean,self).__init__()
        #Initialize the parameters
        self.attn_type = attn_type
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.shared_embedding_size = shared_embedding_size
        self.dropout_im_emb=0.0
        self.dropout_txt_emb=0.0
        self.activation_vse = activation_vse

        #initialize the image emebedding layer
        self.im_embedding = nn.Linear(self.im_size,self.shared_embedding_size)
        if self.dropout_im_emb > 0:
            self.im_embedding_dropout = nn.Dropout(self.dropout_im_emb)
        #initialize the text embedding layer
        self.text_embedding = nn.Linear(self.hidden_size,self.shared_embedding_size)
        if self.dropout_txt_emb > 0:
            self.txt_embedding_dropout = nn.Dropout(self.dropout_txt_emb)

    def forward(self,im_var,decoder_hiddens,criterion_vse=None,context_mask=None):
        """
            Learn the shared space and compute the VSE Loss
            Input:
                im_var: The image features with size (B, D_im)
                decoder_hiddens: The decoder hidden states for each time step of the decoder. Size is (T, B, H), H is the hidden size, T is the decoder_hiddens. 
                criterion_vse: The criterion to compute the loss.
            Output: 
                loss_vse: The loss computed for the visual-text shared space learning.
        """
        #Initialize the loss
        loss_vse = 0
        #Embed the image fetures to the shared space
        im_emb_vec = self.im_embedding(im_var)

        if self.activation_vse:
            im_emb_vec = F.tanh(im_emb_vec)
        
        if self.dropout_im_emb > 0:
            im_emb_vec = self.im_embedding_dropout(im_emb_vec)

        #Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        """
        #Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec,decoder_hiddens,ctx_mask=context_mask)

        context_vec = attn_weights.bmm(decoder_hiddens.transpose(0,1)).squeeze(1)
        """
        #Define the representation of the hidden states as the mean
        context_vec = torch.mean(decoder_hiddens,dim=0) #The size is (B,H)

        text_emb_vec = self.text_embedding(context_vec)
        #Check if apply activateion function on the embedding vector
        if self.activation_vse:
            text_emb_vec = F.tanh(text_emb_vec)

        if self.dropout_txt_emb > 0:
            text_emb_vec = self.txt_embedding_dropout(text_emb_vec)
        #Apply l2 norm to the text_emb_vec
        text_emb_vec = l2norm(text_emb_vec)


        #Compute the loss
        loss_vse = criterion_vse(im_emb_vec,text_emb_vec)

        return loss_vse
        
    def get_emb_vec(self,im_var,decoder_hiddens,ctx_mask=None):
        #Embed the image fetures to the shared space
        im_emb_vec = self.im_embedding(im_var)
        if self.activation_vse:
            im_emb_vec = F.tanh(im_emb_vec)
        #Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        #Compute the Average
        context_vec = torch.mean(decoder_hiddens,dim=0) #The size is (B,H)

        text_emb_vec = self.text_embedding(context_vec)
        if self.activation_vse:
            text_emb_vec = F.tanh(text_emb_vec)
        
        #Apply l2 norm to the text_emb_vec
        text_emb_vec = l2norm(text_emb_vec)

        return im_emb_vec, text_emb_vec

    def get_imagine_weights(self,im_feats,decoder_hiddens):
        raise NotImplementedError