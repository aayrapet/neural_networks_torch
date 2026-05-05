import torch 
import torch.nn as nn 
from ..attention import TransformerBlock

#https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
class TransformerRes(nn.Module):
        def __init__(self,embed,dmodel,n_heads):
            """
            input image is B,C,H,W
            """
            super().__init__()
            self.op=nn.Sequential(TransformerBlock(embed,dmodel,n_heads))
        def forward(self, x):
            return self.op(x)+x


class MLPRes(nn.Module):
    def __init__(self,embed_in,embed_out):
        super().__init__()
        self.op=nn.Sequential(nn.Linear(embed_in,embed_out),nn.GELU(),nn.Linear(embed_out,embed_in))
    def forward(self,x):
        return self.op(x)+x


class TransformerBlock(nn.Module):
    def __init__(self,embed_in,dmodel,n_heads, embed_out):
        super().__init__()
        self.op=nn.Sequential(TransformerRes(embed,dmodel,n_heads),nn.LayerNorm(embed),MLPRes(embed,embed_out),nn.LayerNorm(embed))

    def forward(self,x):
        return self.op(x)

class TextEncoder(nn.Module):
    def __init__(self,embed_in,dmodel,n_heads, T,embed_out,token_size,N_blocks):
        self.token_embed=nn.Linear(VOCAB_SIZE,embed_in)

   

        self.position_embed=nn.Parameter(torch.randn(1,T,embed_in))
        self.operations=nn.ModuleList([TransformerBlock(embed_in,dmodel,n_heads, embed_out) for _ in range( N_blocks)])

        self
    def forward(self, x):
        x=self.token_embed(x)+self.position_embed
        for i in range(len(self.operations )):
            x=self.operations[i](x)
        return x@self.token_embed.weight.T

