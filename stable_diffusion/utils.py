import torch 
import torch.nn as nn 
import torch.nn.functional as F

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

class Normalisation(nn.Module):
    def __init__(self,channels,default_groups=32):
        super().__init__()
        nb_groups=channels if channels% default_groups !=0 else default_groups
        self.norm=nn.GroupNorm(nb_groups,channels)
    def forward(self,x):
        return self.norm(x)

