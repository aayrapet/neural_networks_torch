import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math as mt 

            

class SpatialTransformer(nn.Module):
    #we take image N C H W 4D !!!!
    #and return image N C H W 4D !!!! 
    def __init__(self, channels,d_cond,d_attn,n_heads,N_transformers,flash=False):

        default_groups=32
        nb_groups=channels if channels% default_groups !=0 else default_groups

        super().__init__()
        self.conv1=nn.Conv2d(channels,channels,1,1,0)
        self.conv2=nn.Conv2d(channels,channels,1,1,0)
        self.GN32=nn.GroupNorm(nb_groups,channels)
        #based on out notation, d_model=channels easily 
        self.transformer_blocks=nn.ModuleList([TransformerBlock(channels,d_cond,d_attn,n_heads,flash) for _ in range(N_transformers)])
    def forward(self,x,cond):

        #x is our image representation in 4D
        #cond is our conditional embedding representation in 3D (for transformers directly )  [N,  n_cond, d_cond]
        N,C,H,W=x.shape
        x_original=x
        x=self.conv1(self.GN32(x))
        #from 4D to 3D for transformers [N,H*W,C] which in our notation [B,T,dmodel]
        x=x.reshape(N,C,H*W).transpose(2,1)

        for transformer in self.transformer_blocks:
            x=transformer(x,cond)
        x=x.transpose(1,2).reshape(N,C,H,W)
        x=self.conv2(x)+x_original
        return x 

class TransformerBlock(nn.Module):
    def __init__(self, d_model,d_cond,d_attn,n_heads,flash=False):
        super().__init__()
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        #self attention only on x which last dim is dmodel
        self.SA=CrossAttention( d_model,d_model,d_attn,n_heads,flash)
        #cross attention on x and cond
        self.CA=CrossAttention( d_model,d_cond,d_attn,n_heads,flash)
        self.FF=FeedForward(d_model)
    def forward(self,x,cond):
        x=self.SA(self.norm1(x))+x
        x=self.CA(self.norm2(x),cond)+x
        x=self.FF(self.norm3(x))+x
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model,d_cond,d_attn,n_heads,flash=False):
        super().__init__()
        if d_attn%n_heads!=0:
            raise ValueError("d_attn must be divisibale by n_heads")
        self.Q = nn.Linear(d_model, d_attn, bias=False)
        self.K = nn.Linear(d_cond, d_attn, bias=False)
        self.V = nn.Linear(d_cond, d_attn, bias=False)
        small_size=int(d_attn/n_heads)
        self.small_size = small_size
        self.n_heads=n_heads
        self.linear=nn.Linear(d_attn,d_model)
        self.flash =None
        if flash:
                from flash_attn.flash_attention import FlashAttention
                self.flash = FlashAttention()
                self.flash.softmax_scale = self.scale


    def forward(self, x,cond=None):
        #x, cond are 3D : B,T,C
        if cond is None:
            cond=x
        q = self.Q(x)
        k = self.K(cond)
        v = self.V(cond)
        if self.flash is not None:
            return self.flash_attention(q,k,v) 
        return self.attention(q, k, v)
    
    def flash_attention(self,q,k,v):
        #now we do multihead attention : split big Q,K,V of size [B,T,d_attn]
        qkv=torch.stack((q,k,v),dim=-1)#[B,T,d_attn*3]
        qkv=qkv.view(*qkv.shape[0:2],3,self.n_heads,self.small_size)#[B,T,3,d_attn/n_heads,n_heads]
        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.self.small_size <= 32:
            pad = 32 - self.small_size
        elif self.self.small_size <= 64:
            pad = 64 - self.small_size
        elif self.small_size <= 128:
            pad = 128 - self.small_size
        else:
            raise ValueError(f'Head size ${self.small_size} too large for Flash Attention')
        
        if pad:
            qkv=torch.cat(qkv,torch.zeros((*qkv.shape[0:2],3,self.n_heads,pad)),dim=-1)
   
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, :self.small_size]
        out=out.reshape(*qkv.shape[0:2],self.small_size*self.n_heads)

        return self.linear(out)#from B,T,d_attn to  B,T,d_model

    def attention(self,q,k,v):
        #now we do multihead attention : split big Q,K,V of size [B,T,d_attn]
        #into [B,T,d_attn/n_heads,n_heads] simply wich means qi,ki,vi of size [B,T,d_attn/n_heads] for i =1....n_heads

        #small size is d_attn/n_heads, the rest is n_heads automatically
        q=q.view(*q.shape[0:2],self.small_size,-1)
        k=k.view(*k.shape[0:2],self.small_size,-1)
        v=v.view(*v.shape[0:2],self.small_size,-1)

        #do qi@ki.T for all i=1...n_heads divided by sqrt
        attn =torch.einsum("bidn,bjdn->bijn", q, k)*self.small_size**-0.5
        #softmax over qi@ki.T all i over d_attn/n_heads dim, which is second last dim
        attn=attn.softmax(dim=-2)
        #easily multiply softmax by v 
        out=torch.einsum("bijn,bjkn->bikn", attn, v)

        #stack all i together to get back tensor dim [B,T,d_attn]

        out=out.reshape(*out.shape[0:2],-1)

        #[B,T,d_model]

        return self.linear(out)

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            #will be n_embd * 4*2 then /2 so back to n_embd * 4
            GEGLU(n_embd, n_embd * 4),
            nn.Dropout(0.),
            nn.Linear(n_embd * 4, n_embd)
        )
        
    def forward(self, x):
        out = self.net(x)
        return out
    
class GEGLU(nn.Module):
    """
    A variant of the gated linear unit activation function 
    from https://arxiv.org/abs/2002.05202.

    https://gist.github.com/hskim-solv/02b0782b56b2219ff9485f7baae5de59
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate): # not support mps        
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, h):
        h, gate = self.proj(h).chunk(2, dim=-1)
        return h * self.gelu(gate)


if __name__=="__main__":

    #test spacial transformer 
    N=1
    d_cond=2
    X=torch.randn((N,3,24,24))

    cond=torch.randn((N,4,d_cond))
    print("original matrix shape", X.shape)
    model=SpatialTransformer(3,d_cond,6,3,2)
    print("one forward pass  matrix shape", model(X,cond).shape)


    #test attention 

    model2=CrossAttention(d_cond,d_cond,6,3)
    print("one forward pass  attention matrix shape", model2(cond).shape)


    cond2=torch.randn((N,4,5))
    model3=CrossAttention(d_cond,5,6,3)

    print("one forward pass  cross attention matrix shape", model3(cond,cond2).shape)


    #test attention

