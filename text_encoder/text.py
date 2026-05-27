import torch 
import torch.nn as nn 
import sys 
import os
import torch.nn.init as init
print( sys.path[0])
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
print( sys.path[0])
from  attention import  AttentionBlock

#https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
class TransformerRes(nn.Module):

        
        def __init__(self,embed,dmodel,n_heads,nb_tokens,masked):

            super().__init__()
            self.op=nn.Sequential(AttentionBlock(embed,dmodel,n_heads,nb_tokens,masked))


        def forward(self, x):
            return self.op(x)+x


class MLPRes(nn.Module):
    def __init__(self,embed_in,embed_out):
        super().__init__()
        self.op=nn.Sequential(nn.Linear(embed_in,embed_out),nn.GELU(),nn.Linear(embed_out,embed_in))
    def forward(self,x):
        return self.op(x)+x

import math as mt 
class TransformerBlock(nn.Module):
    def __init__(self,embed_in,dmodel,n_heads,nb_tokens,masked, embed_out):
        super().__init__()
        #this chain is done in attention is all you need and https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
        # self.op=nn.Sequential(TransformerRes(embed_in,dmodel,n_heads,nb_tokens,masked),nn.LayerNorm(embed_in),MLPRes(embed_in,embed_out),nn.LayerNorm(embed_in))
        #we use this for clip text encoder instead : https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        self.op=nn.Sequential(nn.LayerNorm(embed_in),TransformerRes(embed_in,dmodel,n_heads,nb_tokens,masked),nn.LayerNorm(embed_in),MLPRes(embed_in,embed_out))


    def forward(self,x):
        return self.op(x)

class TextEncoder(nn.Module):
    def __init__(self,nb_tokens,VOCAB_SIZE,embed_in,dmodel,n_heads,embed_out,N_blocks,masked=True):
        super().__init__()
        self.token_embed=nn.Embedding(VOCAB_SIZE,embed_in)
        self.position_embed=nn.Parameter(torch.randn(1,nb_tokens,embed_in))
        self.operations=nn.ModuleList([TransformerBlock(embed_in,dmodel,n_heads,nb_tokens,masked, embed_out) for _ in range( N_blocks)])
        #https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf again ,(2.3 Model)
        self.operations.append(nn.LayerNorm(embed_in))

        self.nb_residual_layers=N_blocks*2 #inside TransformerBlock there is transformerres block and mlpres block 
        self.operations.apply(self.weights_scaling)

    def weights_scaling(self,m):
            if isinstance(m, nn.Linear):
                m.weight.data.mul_(1/mt.sqrt(self.nb_residual_layers)) 


    def forward(self, x):
        """
        x: 2D matrix : size  N,nb_tokens 
        note: x is 2D matrix that we get after BPE tokenization 

        """
        
        x=self.token_embed(x)#we get N,nb_tokens,embed_in 3D tensor 
        x=x+self.position_embed#add positional embed simply 


        for i in range(len(self.operations )):
            x=self.operations[i](x)
        # output= x@self.token_embed.weight.T
        eos_token=x[:,-1,:]
        return eos_token
        #eos token (since mask)


if __name__=="__main__":
    N=100
    nb_tokens_per_doc=50#context size this is 
    #as if we use BPE to get 2D tensor
    VOCAB_SIZE=1000
    embed_in=200#one token is represented by  200 digits
    dmodel=300
    n_heads=10
    embed_out=embed_in*2
    N_blocks=12
    masked=True


    x=torch.randint(VOCAB_SIZE,(N,nb_tokens_per_doc))

    m=TextEncoder(nb_tokens_per_doc,VOCAB_SIZE,embed_in,dmodel,n_heads,embed_out,N_blocks,masked)
    print(m(x).shape)
    print(m)
    #should be N,nb_tokens_per_doc,VOCAB_SIZE