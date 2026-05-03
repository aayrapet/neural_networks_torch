import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math as mt 
from utils import Upsample,Downsample,Normalisation
from transformer import SpatialTransformer



class ResBlock(nn.Module):
    def __init__(self,channels,d_t_emb,out_channels):
        super().__init__()
        
        self.norm1=Normalisation(channels)
        self.norm2=Normalisation(out_channels)
        self.act1=nn.SiLU()
        self.act2=nn.SiLU()
        self.act3=nn.SiLU()
        self.conv1=nn.Conv2d(channels,out_channels,3,1,1)
        self.lin=nn.Linear(d_t_emb,out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,1,1)

        if out_channels==channels:
            self.skip_connection=nn.Identity()
        else:
            self.skip_connection=nn.Conv2d(channels,out_channels,1,1)

    def forward(self,x,t):
        #t of shape [B, d_t_emb]
        #x of shape [B, channels,H,W]
        x_original=x
        x=self.conv1(self.act1(self.norm1(x)))
        t=self.lin(self.act2(t))
        #pass t 2D into 4D to sum with x which is image 4D
        t=t.unsqueeze(-1).unsqueeze(-1)
        h=x+t
        h=self.conv2(self.act3(self.norm2(h)))

        return self.skip_connection(x_original)+h
   

class SequentialCustom(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules such as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, cond,t):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x
class Unet(nn.Module):
    def __init__(self,channels_in,
                 channels,output_channels: list,
                 out_channels,
                 n_res_blocks ,
                 indexes_transformers : list ,
                 d_cond,
                 d_attn,
                 n_heads,
                 N_transformers):
        """ 
        channels_in channels of image x input

        channels is output channels of first convolution 
        output_channels is list of channels outputs in convolutions/transformers
        indexes_transformers are indexes from output_channels when transformers are needed, can be empty 
        d_cond #cond of shape [B, T,d_cond]
        other params are coming from transformers and just fixed
        
        """
        super().__init__()
        self.channels = channels
        challes_output_dtembed=channels*4
        self.t_embed_op=nn.Sequential(nn.Linear(channels,challes_output_dtembed),
                                      nn.SiLU(),
                                      nn.Linear(challes_output_dtembed,challes_output_dtembed))

        encoder=nn.ModuleList([])
        encoder.append(SequentialCustom(nn.Conv2d(channels_in,channels,3,1,1)))
        input_block_channels=[channels]
        for i in range(len(output_channels)):

            for _ in range(n_res_blocks):
                chain=[]

                chain.append(ResBlock(channels,challes_output_dtembed,output_channels[i]))
                channels=output_channels[i]
                
                if i in indexes_transformers:
                    chain.append(SpatialTransformer(channels,d_cond,d_attn,n_heads,N_transformers))
                encoder.append(SequentialCustom(*chain))
                input_block_channels.append(channels)

            if i<len(output_channels)-1:
                    encoder.append(SequentialCustom(Downsample(channels)))
                    input_block_channels.append(channels)

        self.encoder=encoder
        self.mid_block=SequentialCustom(ResBlock(channels,challes_output_dtembed,channels),SpatialTransformer(channels,d_cond,d_attn,n_heads,N_transformers),
                        ResBlock(channels,challes_output_dtembed,channels))

        decoder=nn.ModuleList([])
        for i in range(len(output_channels)-1,-1,-1):

            #trick +1 to keep l*(r-1) operations
            #so if nresblocks=1 repeat them 2 times 
            for j in range(n_res_blocks+1):
                chain=[]

                chain.append(ResBlock(channels+input_block_channels.pop(),challes_output_dtembed,output_channels[i]))
                channels=output_channels[i]
                if i in indexes_transformers:
                    chain.append(SpatialTransformer(channels,d_cond,d_attn,n_heads,N_transformers))
                

                if i>0 and j==n_res_blocks :
                    chain.append(Upsample(channels))
                decoder.append(SequentialCustom(*chain))
        self.decoder=decoder
        self.final=nn.Sequential(
            Normalisation(channels),
            nn.SiLU(),
            nn.Conv2d(channels,out_channels,3,1,1)
        )

  
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
            """
            ## Create sinusoidal time step embeddings


            :param time_steps: are the time steps of shape `[batch_size]`
            :param max_period: controls the minimum frequency of the embeddings.

            output:

            matrix [batch_size,channels]
            """
            #  half the channels are sin and the other half is cos,
            half = self.channels // 2
            #freq i = (1/ max) ^(i/half) written in exp way 
            frequencies = torch.exp(
                -mt.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=time_steps.device)
            # $\frac{t}{10000^{\frac{2i}{c}}}$
            #mult vector [batch_size,1] by [1,half] to get matrix [batch_size,half]
            #multiply each time stemp by frequencies which are multiplicators within sin(t*cst) to strengthen waves 
            args = time_steps[:, None].float() * frequencies[None]
            
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self,x,cond,t):
        #t of shape [B ]
        #x of shape [B, channels_in,H,W]
        #cond of shape [B, T,d_cond]
        t_emb = self.time_step_embedding(t)
        t_emb=self.t_embed_op(t_emb) #becomes [B, challes_output_dtembed]
        store_encoder=[]

        #encoder part 
        for op in self.encoder:
            x=op(x,cond,t_emb)
            store_encoder.append(x)
        #mid part 
     

        x=self.mid_block(x,cond,t_emb)
        for op in self.decoder:
            x=torch.concatenate([x,store_encoder.pop()],dim=1)
            x=op(x,cond,t_emb)
        
        return self.final(x)
    

if __name__=="__main__":
    model=Unet(channels_in=32,
                    channels=32,output_channels= [64],
                    out_channels=32,
                    n_res_blocks=1 ,
                    indexes_transformers=[] ,
                    d_cond=2,
                    d_attn=2,
                    n_heads=2,
                    N_transformers=2)
    t= torch.arange(0, 10)
    x = torch.randn(10, 32, 20, 20)
    cond = torch.randn(10, 3, 2)

    print(
    model(x,cond,t).shape)


        