## Defines a Transformer Encoder Architecture for Denoising Network, inspired from Human Motion Diffusion Models(https://arxiv.org/abs/2209.14916)


import torch
import numpy as np 
import torch.nn as nn
import math

class time_embedding(nn.Module):
    def __init__(self,n_hid_dim):
        super(time_embedding, self).__init__() 
        self.layers = nn.Sequential(*[
            nn.Linear(1, n_hid_dim),
            nn.ReLU(),
            nn.Linear(n_hid_dim, n_hid_dim),
            nn.ReLU(),
        ])

    def forward(self, t):
        time_embds = self.layers(t)
        return time_embds
    

# This will implement two fc layers as a position wise feed forward network placed inside the attention block 
class position_wise_ff(nn.Module):
    def __init__(self, inter_dim, h_dim, dropout):
        super(position_wise_ff, self).__init__()
        self.linear1 = nn.Linear(h_dim, inter_dim)
        self.linear2 = nn.Linear(inter_dim, h_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = nn.GELU()(x) 
        x = self.linear2(x)
        x = self.dropout2(x) 

        return x 


# This is a transformer block from "Attention is all you need paper". To create an encoder multiple copies of this attention block will be used 
class attention_block(nn.Module):
    def __init__(self, h_dim, n_heads, dropout):
        super(attention_block, self).__init__()
        self.multi_head = nn.MultiheadAttention(h_dim, n_heads)
        self.ff = position_wise_ff(h_dim * 4, h_dim, dropout) # Keeping the interdim as 4 * h_dim | dropout will already be in the ff network 
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.dropout_mha = nn.Dropout(dropout)

    def forward(self, x):
        # print("Inside attention module ...... ")
        # print("x type mha: {}".format(x.get_device()))
        
        mh_x, _ = self.multi_head(x,x,x) # computing the self attention with the same input vector as q,k and v
        mh_x = self.dropout_mha(mh_x)
        x = self.ln1(x + mh_x)      # skip connection + normalization

        ff_x = self.ff(x) # fead forward layer 
        x = self.ln2(x + ff_x)      # skip connection + normalization 
        return x 

class transformer_encoder(nn.Module):              
    def __init__(self, condition_dim, out_dim, hidden_dim, H, device):
        super(transformer_encoder, self).__init__() 
        self.condition_dim = condition_dim
        self.inp_dim = out_dim
        self.out_dim = out_dim
        self.h_dim = hidden_dim
        self.n_heads = 8
        self.n_blocks = 6
        self.pos_enc_dim = hidden_dim
        self.dropout = 0.1
        self.device = device
        self.H = H

        self.proj_obs_condition = nn.Linear(self.inp_dim, self.h_dim)  # condition : 7 to 128
        self.proj_emb_condition = nn.Linear(condition_dim - self.inp_dim, self.h_dim)  # condition : 2048 to 128
        self.proj_layer = nn.Linear(self.inp_dim, self.h_dim)  
        # This is a small mlp that will project the time index to the hidden dimension which is input of the model 
        self.time_embedder = time_embedding(self.h_dim).to(device)

        self.attn_blocks = nn.ModuleList([attention_block(self.h_dim, self.n_heads, self.dropout).to(device) for _ in range(self.n_blocks)])

        # This is the inverse projection layer that will map the processed vector into the original dimensional space for the input 
        self.inverse_proj_layer = nn.Linear(self.h_dim, self.out_dim) 

        print("Transformer encoder model: {}".format(self.modules))

    # This function will compute the positional encodings of the source timestep sequence with the given dimensions 
    def pos_encs(self,timesteps,dim,max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                        torch.arange(start=0, end=half, dtype=torch.float32) /
                        half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.device)
        embedding = embedding.squeeze(1) # Removing the extra channel with single dimension 
        # print("embds shape: {}".format(embedding.shape))           
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding 

    # This function will perform forward pass through the transformer encoder with all the modules and will predict the concatenated output embeddings 
    def forward(self, x, t, condition):

        ## Input x - (B, 16, 7)
        ## Input t - (B, 1)
        ## Input condition - (B, 2055)

        B, N = x.shape[0], x.shape[1] # B: batch size, N: number of tokens

        obs_cond = condition[:,:7]
        emb_cond = condition[:,7:]

        obs_cond = self.proj_obs_condition(obs_cond).unsqueeze(1)
        emb_cond = self.proj_emb_condition(emb_cond).unsqueeze(1)

        x = self.proj_layer(x) # B x N x 7 -> B x N x h_dim

        idx = torch.Tensor([id for id in range(1, N+1)]) # idx = [1, ... N] 
        pos_enc = self.pos_encs(idx, self.pos_enc_dim) # N -> N x h_dim 
        x = x + pos_enc # B x N x h_dim -> B x N x h_dim
        time_embds = self.time_embedder(t).reshape(B, 1, self.h_dim) # B x h_dim -> B x 1 x h_dim
                
        x = torch.cat([obs_cond, emb_cond, time_embds, x], axis=1) # B x N x h_dim -> B x (N+3) x h_dim 

        # Looping over the attention layers to obtain the transformed encoder from the latent space 
        for idx in range(0, self.n_blocks):
            x = self.attn_blocks[idx](x)

        # As the first token was for the time embeddings it is not part of the predicted sequence, hence we will use the sequence after that 
        seq_chop = x[:,3:,:] 


        seq_unproj = self.inverse_proj_layer(seq_chop) # B x N x h_dim -> B x N x inp_dim
    
        return seq_unproj

# The function to test the implementation of the attention model used 
def test_main():
    
    model = transformer_encoder(condition_dim=2055, out_dim=7, hidden_dim=128, H=16,device='cuda').to('cuda')

    x_inp = torch.randn(8, 16, 7).to('cuda')
    condition = torch.randn(8, 2055).to('cuda')
    t = torch.randn(8,1).to('cuda')

    y = model(x_inp,t, condition)
    print("input x shape: {}, t: {}".format(x_inp.shape, t))
    print("model output shape: {}".format(y.shape))

if __name__ == "__main__":
    test_main()