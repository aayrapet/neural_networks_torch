import torch 
import torch.nn as nn 
from transformer import CrossAttention
import torch.nn.functional as F

from utils import Normalisation,swish,Downsample,Upsample
class ResBlock(nn.Module):
    def __init__(self,channels,out_channels):
        super().__init__()
        
        self.norm1=Normalisation(channels)
        self.norm2=Normalisation(out_channels)
    
        self.conv1=nn.Conv2d(channels,out_channels,3,1,1)
      
        self.conv2=nn.Conv2d(out_channels,out_channels,3,1,1)

        if out_channels==channels:
            self.skip_connection=nn.Identity()
        else:
            self.skip_connection=nn.Conv2d(channels,out_channels,1,1)

    def forward(self,x):
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
                print('e',channels,'-',output_channels[i+1])
                channels=output_channels[i+1]
            if i==K-1:
                chain.append(nn.Identity())
                print('I')
            else:
                chain.append(Downsample(channels))
                print('D')

        self.down=nn.Sequential(*chain)

       
        self.norm1=Normalisation(channels)

        
        self.midres1=ResBlock(channels,channels)
        #norm1 first 
        #perform transformation from 4d to 3D B,H*W,C
        self.midattn=CrossAttention(channels,channels,channels,1)
        # Reshape back to `[batch_size, channels, height, width]`
        self.midres2=ResBlock(channels,channels)

        self.norm2=Normalisation(channels)
        self.conv_out = nn.Conv2d(channels, z_channels, 3, stride=1, padding=1)
        print(channels,"-",z_channels)
        
    def forward(self,x):
   
        x=self.conv(x)
        x=self.down(x)
        x=self.midres1(x)
        x=self.norm1(x)
        B,C,H,W=x.shape
        x=x.permute(0,2,3,1)#important for every reshape
        x=x.reshape(B,H*W,C)
        print(x.shape)
        x=self.midattn(x)
        print(x.shape)

        x=x.reshape(B,H,W,C)
        x=x.permute(0,3,1,2)
        x=self.midres2(x)
        x=self.norm2(x)
        x=swish(x)
        x=self.conv_out(x)
        """
        output is B,z_channels,Hz,Wz tensor 
        
        """
        return x
    

class Decoder(nn.Module):
    def __init__(self,channels, output_channels,channels_f,n_res_blocks,z_channels,out_channels):
        super().__init__()

        self.conv_in = nn.Conv2d(z_channels,channels, 3,stride=1, padding=1)
        print(z_channels,"-",channels)
        self.norm1=Normalisation(channels)
        
        self.midres1=ResBlock(channels,channels)
        self.midattn=CrossAttention(channels,channels,channels,1)
        self.midres2=ResBlock(channels,channels)



        K=len(output_channels)
        output_channels = output_channels[::-1]
        output_channels=output_channels+[channels_f]

        chain=[]
        for i in range(K):

            for j in range(n_res_blocks):

                if j==n_res_blocks-1:
                    next=output_channels[i+1]
                else:
                    next=channels
                chain.append(ResBlock(channels,next))
                print('d',channels,'-',next)
                if j==n_res_blocks-1:
                    channels=output_channels[i+1]

            if i==K-1:
                chain.append(nn.Identity())
                print("I")
            else:
                chain.append(Upsample(channels))
                print("D")


        self.up=nn.Sequential(*chain)

       
        self.norm2=Normalisation(channels)

        

        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)
        print(channels,"-",out_channels)

    def forward(self,x):

        x=self.conv_in(x)
        x=self.midres1(x)
        x=self.norm1(x)
        B,C,H,W=x.shape
        x=x.permute(0,2,3,1)
        x=x.reshape(B,H*W,C)
        x=self.midattn(x)
        x=x.reshape(B,H,W,C)
        x=x.permute(0,3,1,2)
        x=self.midres2(x)
        x=self.up(x)
        x=self.norm2(x)
        x=swish(x)
        x=self.conv_out(x)
        return x 





class VAE(nn.Module):
    def __init__(self,channels_in,channels,output_channels,n_res_blocks,z_channels,emb_channels):
        print('remember to use powers of 2 : 32,62,128..., otherwise will be cropped ')
        super().__init__()
        self.encoder=Encoder(channels_in,channels,output_channels,n_res_blocks,z_channels)
        self.transformation_before_sampling=nn.Conv2d(z_channels,2*emb_channels,1,1,0)
        self.sampler=GaussianSampling()
        self.transformation_after_sampling=nn.Conv2d(emb_channels,z_channels,1,1,0)
        self.decoder=Decoder(output_channels[-1],output_channels,channels,n_res_blocks,z_channels,channels_in)

    def forward(self,x):
        B,C,H,W=x.shape
        x=self.encoder(x)
        x=self.transformation_before_sampling(x)
        x,mean,log_var=self.sampler(x)
        x=self.transformation_after_sampling(x)
        x=self.decoder(x)
        return x[:,:,:H,:W],mean,log_var

        
class VaeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,decoded_image,input_image,mean,log_var):
        
        #sum over dim of hidden representation
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(),dim=(1, 2, 3))
        kl_div = torch.mean(kl_div)#over all images do mean over batch 
        recons_loss = F.mse_loss(decoded_image, input_image)
        loss=kl_div+recons_loss
        return loss 
    

class GaussianSampling(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,embed):
        #divide in 2 parts mean and logvar 
        mean,log_var=embed.chunk(2,dim=1)

        sigma=torch.exp(0.5*log_var)
        epsilon=torch.randn_like(sigma)
        return mean+sigma*epsilon,mean,log_var
    




if __name__=="__main__":
    x=torch.randn(10,3,300,300)
    m=VAE(channels_in=3,channels=4,output_channels=[5,6,7,8],n_res_blocks=5,z_channels=2,emb_channels=2)
    print(m(x)[0].shape)
