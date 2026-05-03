import torch 
import torch.nn as nn 
from attention import TransformerBlock
class ViT(nn.Module):
    def __init__(self,H,W,C,D,N,dmodel,n_heads,num_classes,patch_resolution=16):
        """
        input image is B,C,H,W
        """
        super().__init__()

        self.ptf=PatchTransformation(H,W,C,patch_resolution=patch_resolution)
        self.linear=nn.Linear(self.ptf.embed,D)
        self.cls_token=nn.Parameter(torch.randn(1,1,D))
        self.pos_embed=nn.Parameter(torch.randn(1,self.ptf.nb_patches+1,D))
        self.blocks=nn.ModuleList([ Block(D,dmodel,n_heads) for _ in range(N )])

        self.final_linear=nn.Linear(D,num_classes)
        self.finalnorm=nn.LayerNorm(D)

    def forward(self,x):
        B,C,H,W=x.shape
        patches=self.ptf(x)
        patch_embed=self.linear(patches)#B,N,D
        cls_token=self.cls_token.expand(B,-1,-1)
        patch_embed_cls=torch.cat([cls_token,patch_embed],dim=1)#B,N+1,D
        x=patch_embed_cls+self.pos_embed#B,N+1,D
        
        for i in range(len(self.blocks)):
            x=self.blocks[i](x)
        
        extracted_cls_token=x[:,0,:]#in VIT we just extract first cls token that summaries the image so the size is simple B,D 
        extracted_cls_token=self.finalnorm(extracted_cls_token)
        return self.final_linear(extracted_cls_token)

class Block(nn.Module):
    def __init__(self,embed,dmodel,n_heads):
        """
        input image is B,C,H,W
        """
        super().__init__()
        self.op=nn.Sequential(
            TransformerBlockRes(embed,dmodel,n_heads),
        MLPRes(embed,embed*2))

    def forward(self,x):
        return self.op(x)

class TransformerBlockRes(nn.Module):
        def __init__(self,embed,dmodel,n_heads):
            """
            input image is B,C,H,W
            """
            super().__init__()
            self.op=nn.Sequential(nn.LayerNorm(embed),TransformerBlock(embed,dmodel,n_heads))
        def forward(self, x):
            return self.op(x)+x

class MLPRes(nn.Module):
    def __init__(self,embed_in,embed_out):
        super().__init__()
        self.op=nn.Sequential(nn.LayerNorm(embed_in),nn.Linear(embed_in,embed_out),nn.GELU(),nn.Linear(embed_out,embed_in))
    def forward(self,x):
        return self.op(x )+x

class PatchTransformation(nn.Module):
    """
    Image is worth 16*16 words
    """
    def __init__(self,H,W,C,patch_resolution=16):
        """
        input image is B,C,H,W
        """
        super().__init__()

        if H%patch_resolution != 0 or W%patch_resolution!= 0:
            raise ValueError("H, W have to be divisible by patch resolution ")
        self.nb_patches=int(H*W/patch_resolution**2)
        self.patch_resolution=patch_resolution
        self.area=patch_resolution**2
        self.embed=self.area*C
        self.H=H
        self.W=W
        self.C=C

    def forward(self,x):

        if x.ndim!=4:
            raise ValueError("has to be 4D tensor ")
        
        B,C,H,W=x.shape
        if self.C!=C or self.W!=W or self.H!=H:
             raise ValueError("de facto values hwc dont match with decalred ")
        
        patches=x.unfold(2,size=self.patch_resolution,step=self.patch_resolution).unfold(3,size=self.patch_resolution,step=self.patch_resolution)

        patches=patches.permute(0,2,3,4,5,1)
        patches=patches.reshape(B,self.nb_patches,self.embed)
        #N is nb of patches, embed is patch area*c
        return patches #B,N,embed
    
if __name__=="__main__":
    #test spacial transformer 
    N=1
    
    X=torch.randn((N,3,224,224))

    m=ViT(224,224,3,5,7,14,7,10)
    print(m(X).shape)
