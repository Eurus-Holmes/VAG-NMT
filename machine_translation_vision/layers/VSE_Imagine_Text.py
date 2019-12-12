#Implement the VSE module such that we directly embed the shared space to text dimension without text_embedding.
import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import l2norm

class ImagineAttn(nn.Module):
    def __init__(self,method,context_size,shared_embedding_size):
        super(ImagineAttn, self).__init__()
        self.method = method
        self.embedding_size = shared_embedding_size
        self.context_size = context_size
        self.mid_dim = self.context_size

        self.ctx2ctx = nn.Linear(self.context_size,self.context_size,bias=False)
        self.emb2ctx = nn.Linear(self.embedding_size,self.context_size,bias=False)
        
        if self.method == 'mlp':
            self.mlp = nn.Linear(self.mid_dim,1,bias=False)
            self.score = self.score_mlp
        if self.method == 'dot':
            self.score = self.score_dot
        '''
        if self.method == 'dot':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        '''
    def forward(self,image_vec,decoder_hidden,ctx_mask=None):
        '''
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(T,B,E)
            context_mask: The mask applied to filter out the hidden states that don't contribute. The size is (T,B)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,T) 
        '''

        #Create variable to store attention energies
        attn_energies = self.score(image_vec.unsqueeze(1),decoder_hidden.transpose(0,1))
        if ctx_mask is not None:
            self.mask = (1-ctx_mask.transpose(0,1).data).byte().unsqueeze(1) #Convert the mask to the size(B*1*T)
            attn_energies.data.masked_fill_(self.mask,-float('inf'))
        #Normalize energies to weights in range 0 to 1, resize to B x 1 x T
        return F.softmax(attn_energies,dim=-1)
    def score_dot(self,image_vec,decoder_hidden):
        """
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,1,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(B,T,C)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,1,T) 
        """
        ctx_ = self.ctx2ctx(decoder_hidden).permute(0,2,1) #  B*T*C -> B*C*T
        im_ = self.emb2ctx(image_vec) # B*1*C

        #Apply the l2norm to ctx and im before comutingt the energies
        #ctx_ = l2norm(ctx_,dim=1)
        #im_ = l2norm(im_,dim=2)

        energies = torch.bmm(im_,ctx_) 
        return energies
    def score_mlp(self,image_vec,decoder_hidden):
        """
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,1,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(B,T,C)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,1,T) 
        """
        ctx_ = self.ctx2ctx(decoder_hidden) #  B*T*C
        im_ = self.emb2ctx(image_vec) # B*1*C

        energies = self.mlp(F.tanh(ctx_+im_)).permute(0,2,1) # B*1*T
        return energies

class VSE_Imagine_Text(nn.Module):
    def __init__(self,attn_type,\
                im_size,\
                hidden_size,\
                shared_embedding_size,\
                dropout_im_emb=0.0,\
                dropout_txt_emb=0.0, \
                activation_vse = True):
        super(VSE_Imagine_Text,self).__init__()
        #Initialize the parameters
        self.attn_type = attn_type
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.shared_embedding_size = shared_embedding_size
        self.dropout_im_emb=0.0
        self.dropout_txt_emb=0.0
        self.activation_vse = activation_vse

        #Initialize the layers
        self.imagine_attn =  ImagineAttn(self.attn_type,self.hidden_size,self.shared_embedding_size)
        
        
        #initialize the image emebedding layer
        self.im_embedding = nn.Linear(self.im_size,self.shared_embedding_size)
        if self.dropout_im_emb > 0:
            self.im_embedding_dropout = nn.Dropout(self.dropout_im_emb)
        
        """
        #initialize the text embedding layer
        self.text_embedding = nn.Linear(self.hidden_size,self.shared_embedding_size)
        if self.dropout_txt_emb > 0:
            self.txt_embedding_dropout = nn.Dropout(self.dropout_txt_emb)
        """
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

        #Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec,decoder_hiddens,ctx_mask=context_mask)

        context_vec = attn_weights.bmm(decoder_hiddens.transpose(0,1)).squeeze(1)
        """
        text_emb_vec = self.text_embedding(context_vec)
        if self.activation_vse:
            text_emb_vec = F.tanh(text_emb_vec)

        if self.dropout_txt_emb > 0:
            text_emb_vec = self.txt_embedding_dropout(text_emb_vec)
        """
        text_emb_vec = context_vec

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
        
        if self.dropout_im_emb > 0:
            im_emb_vec = self.im_embedding_dropout(im_emb_vec)

        #Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        #Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec,decoder_hiddens,ctx_mask=ctx_mask)

        context_vec = attn_weights.bmm(decoder_hiddens.transpose(0,1)).squeeze(1)
        """
        text_emb_vec = self.text_embedding(context_vec)
        if self.activation_vse:
            text_emb_vec = F.tanh(text_emb_vec)
        """
        text_emb_vec = context_vec
        #Apply l2 norm to the text_emb_vec
        text_emb_vec = l2norm(text_emb_vec)

        return im_emb_vec, text_emb_vec

    def get_imagine_weights(self,im_var,decoder_hiddens,ctx_mask=None):
        #Embed the image fetures to the shared space
        im_emb_vec = self.im_embedding(im_var)

        if self.activation_vse:
            im_emb_vec = F.tanh(im_emb_vec)
        
        if self.dropout_im_emb > 0:
            im_emb_vec = self.im_embedding_dropout(im_emb_vec)

        #Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        #Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec,decoder_hiddens,ctx_mask=ctx_mask)

        return attn_weights

