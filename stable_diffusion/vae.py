import torch 
import torch.nn as nn 
from transformer import CrossAttention
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,channels,out_channels):
        super().__init__()


        default_groups=32
        nb_groups=channels if channels% default_groups !=0 else default_groups
        
        self.norm1=Normalisation(channels)
        self.norm2=Normalisation(out_channels)
    
        self.conv1=nn.Conv2d(channels,out_channels,3,1,1)
      
        self.conv2=nn.Conv2d(out_channels,out_channels,3,1,1)

        if out_channels==channels:
            self.skip_connection=nn.Identity()
        else:
            self.skip_connection=nn.Conv2d(channels,out_channels,1,1)

    def forward(self,x,t):
        #x of shape [B, channels,H,W]
        x_original=x
        x=self.conv1(swish(self.norm1(x)))
        x=self.conv2(swish(self.norm2(x)))

        return self.skip_connection(x_original)+x
    
class Encoder(nn.Module):
    def __init__(self,channels_in,channels, output_channels,n_res_blocks,z_channels):
        super().__init__()

        self.conv=nn.Conv2d(channels_in,channels,3,1,1)


        K=len(output_channels)

        output_channels=[channels]+output_channels

        chain=[]
        for i in range(K):

            for _ in range(n_res_blocks):
                chain.append(ResBlock(channels,output_channels[i+1]))
                channels=output_channels[i+1]
            if i==K-1:
                chain.append(nn.Identity())
            else:
                chain.append(Downsample(channels))

        self.down=nn.Sequential(*chain)

       
        self.norm1=Normalisation(channels)

        
        self.midres1=ResBlock(channels,channels)
        #norm1 first 
        #perform transformation from 4d to 3D B,H*W,C
        self.midattn=CrossAttention(channels,channels,channels,1)
        # Reshape back to `[batch_size, channels, height, width]`
        self.midres2=ResBlock(channels,channels)

        self.norm2=Normalisation(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)
        
    def forward(self,x):
   

        x=self.conv(x)
        x=self.down(x)
        x=self.midres1(x)
        x=self.norm1(x)
        B,C,H,W=x.shape
        x=x.reshape(B,H*W,C)
        x=self.midattn(x)
        x=x.reshape(H,C,H,W)
        x=self.midres2(x)
        x=self.norm2(x)
        x=swish(x)
        x=self.conv_out(x)
        """
        output is B,2*z_channels,Hz,Wz tensor 
        
        """
        return x
class Decoder(nn.Module):
    def __init__(self,channels, output_channels,channels_f,n_res_blocks,z_channels,out_channels):
        super().__init__()

        self.conv_in = nn.Conv2d(2 * z_channels,channels 3, stride=1, padding=1)
        
    
        self.midres1=ResBlock(channels,channels)
        self.midattn=CrossAttention(channels,channels,channels,1)
        self.midres2=ResBlock(channels,channels)
        self.norm1=nn.GroupNorm(nb_groups,channels)


        K=len(output_channels)
        output_channels.reverse()
        output_channels=output_channels+[channels_f]

        chain=[]
        for i in range(K):

            for _ in range(n_res_blocks):
                chain.append(ResBlock(channels,output_channels[i+1]))
                channels=output_channels[i+1]
            if i==K-1:
                chain.append(nn.Identity())
            else:
                chain.append(Upsample(channels))

        self.up=nn.Sequential(*chain)

       
        self.norm2=Normalisation(channels)

        self.norm1=Normalisation(channels)

        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self,x):

        x=self.conv_in(x)
        x=self.midres1(x)
        x=self.norm1(x)
        B,C,H,W=x.shape
        x=x.reshape(B,H*W,C)
        x=self.midattn(x)
        x=x.reshape(B,C,H,W)
        x=self.midres2(x)
        x=self.up(x)
        x=self.norm2(x)
        x=swish(x)
        x=self.conv_out(x)
        return x 





class VAE(nn.Module):
    def __init__(self,channels_in,channels,output_channels,n_res_blocks,z_channels,emb_channels):
        super().__init__()
        self.encoder=Encoder(channels_in,channels,output_channels,n_res_blocks,z_channels)
        self.transformation_before_sampling=nn.Conv2d(2*z_channels,2*emb_channels,1,1,0)
        self.sampler=GaussianSampling()
        self.transformation_after_sampling=nn.Conv2d(emb_channels,2*z_channels,1,1,0)
        self.decoder=Decoder(output_channels[-1],output_channels,channels,n_res_blocks,z_channels,channels_in)

    def forward(self,x):
        x=self.encoder(x)
        x=self.transformation_before_sampling(x)
        x=self.sampler(x)
        x=self.transformation_after_sampling(x)
        x=self.decoder(x)
        return x

        
    

class GaussianSampling(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,embed):
        #divide in 2 parts mean and logvar 
        mean,log_var=embed.chunk(2,dim=1)

        sigma=torch.exp(0.5*log_var)
        epsilon=torch.randn_like(sigma)
        return mean+sigma*epsilon
    
class Normalisation(nn.Module):
    def __init__(self,channels,default_groups=32):
        super().__init__()
        nb_groups=channels if channels% default_groups !=0 else default_groups
        self.norm=nn.GroupNorm(nb_groups,channels)
    def forward(self,x):
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self,channels,scale=2):
        super().__init__()

        self.conv=nn.Conv2d(channels,channels,3,1,1)
    def forward(self,x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return  self.conv(x)


class Downsample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.op=nn.Conv2d(channels,channels,3,2,1)
    def forward(self,x):
        return self.op(x)

def swish(x: torch.Tensor):
    """
    ### Swish activation

    $$x \cdot \sigma(x)$$
    """
    return x * torch.sigmoid(x)