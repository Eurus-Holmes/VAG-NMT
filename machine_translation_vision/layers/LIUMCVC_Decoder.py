import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .ff import FF


class Attention(nn.Module):
    """Attention layer for seq2seq NMT."""
    def __init__(self, ctx_dim, hid_dim, att_bottleneck='ctx',
                 att_activ='tanh', att_type='mlp'):
        super().__init__()

        # Get activation function
        self.activ = getattr(F, att_activ)

        # Annotation dimensionality
        self.ctx_dim = ctx_dim

        # Hidden state of the RNN (or another arbitrary query entity)
        self.hid_dim = hid_dim

        # The common dimensionality for inner formulation
        if att_bottleneck == 'ctx':
            self.mid_dim = self.ctx_dim
        elif att_bottleneck == 'hid':
            self.mid_dim = self.hid_dim

        # 'dot' or 'mlp'
        self.att_type = att_type

        # MLP attention, i.e. Bahdanau et al.
        if self.att_type == 'mlp':
            self.mlp = nn.Linear(self.mid_dim, 1, bias=False)
            self.forward = self.forward_mlp
        elif self.att_type == 'dot':
            self.forward = self.forward_dot
        else:
            raise Exception('Unknown attention type {}'.format(att_type))

        # Adaptor from RNN's hidden dim to mid_dim
        self.hid2ctx = nn.Linear(self.hid_dim, self.mid_dim, bias=False)

        # Additional context projection within same dimensionality
        self.ctx2ctx = nn.Linear(self.ctx_dim, self.mid_dim, bias=False)

        # ctx2hid: final transformation from ctx to hid
        self.ctx2hid = nn.Linear(self.ctx_dim, self.hid_dim, bias=False)

    def forward_mlp(self, hid, ctx, ctx_mask=None):
        r"""Computes Bahdanau-style MLP attention probabilities between
        decoder's hidden state and source annotations.

        score_t = softmax(mlp * tanh(ctx2ctx*ctx + hid2ctx*hid))

        Arguments:
            hid(Variable): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Variable): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded positions.

        Returns:
            scores(Variable): A variable of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Variable): A variable of shape `B*H` containing the final
                attended context vector for this target decoding timestep.

        Notes:
            This will only work when `T==1` for now.
        """
        # S*B*C and T*B*C
        ctx_ = self.ctx2ctx(ctx)
        hid_ = self.hid2ctx(hid)

        # scores -> S*B
        scores = self.mlp(self.activ(ctx_ + hid_)).squeeze(-1)

        # Normalize attention scores correctly -> S*B
        # NOTE: We can directly use softmax if no mask is given
        if ctx_mask is None:
            ctx_mask = 1.

        alpha = (scores - scores.max(0)[0]).exp().mul(ctx_mask)
        alpha = alpha / alpha.sum(0)

        # Transform final context vector to H for further decoders
        z_t = self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))
        return alpha, z_t

    def forward_dot(self, hid, ctx, ctx_mask):
        r"""Computes Luong-style dot attention probabilities between
        decoder's hidden state and source annotations.

        Arguments:
            hid(Variable): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Variable): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded timesteps.

        Returns:
            scores(Variable): A variable of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Variable): A variable of shape `B*H` containing the final
                attended context vector for this target decoding timestep.
        """
        # Apply transformations first to make last dims both C and then
        # shuffle dims to prepare for batch mat-mult
        ctx_ = self.ctx2ctx(ctx).permute(1, 2, 0)   # S*B*C -> S*B*C -> B*C*S
        hid_ = self.hid2ctx(hid).permute(1, 0, 2)   # T*B*H -> T*B*C -> B*T*C

        # 'dot' scores of B*T*S
        scores = F.softmax(torch.bmm(hid_, ctx_), dim=-1)

        # Transform back to hidden_dim for further decoders
        # B*T*S x B*S*C -> B*T*C -> B*T*H
        z_t = self.ctx2hid(torch.bmm(scores, ctx.transpose(0, 1)))

        return scores.transpose(0, 1), z_t.transpose(0, 1)

class LIUMCVC_Decoder(nn.Module):
    def __init__(self,output_size, \
                 embedding_size, \
                 hidden_size, \
                 context_size, \
                 dropout_emb= 0.0, \
                 dropout_rnn = 0.0, \
                 dropout_out=0.0, \
                 attn_type='mlp', \
                 bias_zero=True, \
                 tied_emb=False):

        super(LIUMCVC_Decoder,self).__init__()

        #Keep Parameters for Reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size 
        self.context_size = context_size
        self.output_size = output_size
        self.dropout_emb = dropout_emb
        self.dropout_rnn = dropout_rnn
        self.dropout_out= dropout_out
        self.attn_type = attn_type
        self.bias_zero = bias_zero
        self.tied_emb = tied_emb

        #Define layers
        self.embedding=nn.Embedding(output_size,embedding_size,padding_idx=0)
        if self.dropout_emb > 0.0:
            self.embedding_dropout = nn.Dropout(dropout_emb)

        #The Three main components for the cGRU.
        self.attn = Attention(context_size, hidden_size,att_type=self.attn_type)
        self.gru_1 = nn.GRUCell(embedding_size,hidden_size)
        self.gru_2 = nn.GRUCell(hidden_size,hidden_size)

        #Three matrix to generate a intermediate representation tj for final output
        
        self.hid2out = FF(hidden_size, embedding_size, bias_zero=bias_zero, activ='tanh')

        #Output Layer
        self.out = FF(embedding_size, output_size)
        
        if self.dropout_out > 0.0:
            self.output_dropout = nn.Dropout(dropout_out)
        
        #Tied Embedding Matrix not implemented yet
        
        if self.tied_emb:
            print("Embedding is tied")
            self.out.weight = self.embedding.weight
        
    def forward(self,word_input,last_hidden,encoder_outputs,ctx_mask=None):
        '''
        Input:
            word_input: A tensor with size B*1, representing the previous predicted word 
            last_hidden: The hidden state vector from the previous timestep, s_t_1
            encoder_outputs: Size T_in*B*Context_Size
        '''
        batch_size = word_input.size()[0]
        #Embedding Word input to WordVectors
        word_embedding = self.embedding(word_input)
        if self.dropout_emb > 0:
            word_embedding = self.embedding_dropout(word_embedding)

        #Process the word_embedding through the first gru to generate the intermediate representation
        gru_1_hidden = self.gru_1(word_embedding,last_hidden) #The gru_1_hidden is the intermediate hidden state, with size(L,B,N), N is the hidden_size, L is the layer_size
        
        #Compute the Attentional Weights Matrix
        alpha_t, z_t = self.attn(gru_1_hidden.unsqueeze(0),encoder_outputs,ctx_mask=ctx_mask)

        #Compute the output from second gru. 
        gru_2_hidden = self.gru_2(z_t,gru_1_hidden)

        logit = self.hid2out(gru_2_hidden)
        #concat_output = F.tanh(self.W1(gru_2_output)) #Adjust this once more with respect to OZAN's implementation.

        if self.dropout_out > 0.0:
            logit =self.output_dropout(logit)

        output = F.log_softmax(self.out(logit),dim=-1)

        return output,gru_2_hidden
