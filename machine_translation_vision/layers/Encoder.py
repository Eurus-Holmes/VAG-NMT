import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math

use_cuda = torch.cuda.is_available()

#Define the Encoder for WMTBaseline_Model
class LIUMCVC_Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size, n_layers=1, dropout_rnn = 0, dropout_emb=0, dropout_ctx=0):
        #Initialize the Super Class
        super(LIUMCVC_Encoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_direction = 2
        #Including the new dropout
        self.dropout_rnn = dropout_rnn
        self.dropout_emb = dropout_emb
        self.dropout_ctx = dropout_ctx
        #Define the Embedding Layer
        self.embedding = nn.Embedding(input_size,embedding_size,padding_idx=0) #Later We want to initialize it with Word2Vec
        
        #Define a Embedding Dropout_Layer, added from LIUMVIC Structure
        if dropout_emb> 0:
            self.embedding_dropout = nn.Dropout(self.dropout_emb)
        
        #Define a source annotation dropou_layey, added from LIUMVIC Structure
        if dropout_ctx > 0:
            self.context_dropout = nn.Dropout(self.dropout_ctx)

        #Define the LSTM Cells
        self.gru = nn.GRU(embedding_size,hidden_size,num_layers=n_layers,bidirectional=True, dropout=self.dropout_rnn)
    
    def forward(self,input_var,input_lengths):
        """
        Input Variable:
            input_var: A variables whose size is (B,W), B is the batch size and W is the longest sequence length in the batch 
            input_lengths: The lengths of each element in the batch. 
            hidden: The hidden state variable whose size is (num_layer*num_directions,batch_size,hidden_size)
        Output:
            output: A variable with tensor size W*B*N, W is the maximum length of the batch, B is the batch size, and N is the hidden size
            hidden: The hidden state variable with tensor size (num_layer*num_direction,B,N)
        """
        #Get the mask for input_var
        ctx_mask = (input_var != 0).long().transpose(0,1)
        
        #Convert input sequence into a pack_padded tensor
        embedded_x = self.embedding(input_var).transpose(0,1) #The dimension of embedded_x is  W*B*N, where N is the embedding size.
        if self.dropout_emb > 0:
            embedded_x = self.embedding_dropout(embedded_x)

        #Get a pack_padded sequence
        embedded_x = torch.nn.utils.rnn.pack_padded_sequence(embedded_x,input_lengths)
        
        #Get an output pack_padded sequence
        output,hidden = self.gru(embedded_x) 
        #Unpack the pack_padded sequence
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output) #The size of output will be W*B*N
        
        #Apply the dropout
        if self.dropout_ctx > 0:
            output = self.context_dropout(output)
        
        return output,ctx_mask.float()