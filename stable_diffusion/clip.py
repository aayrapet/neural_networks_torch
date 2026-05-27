import torch 
import torch.nn as nn 
import sys 
import os
import torch.nn.init as init
print( sys.path[0])
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "vit"))

from  vit import ViT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "text_encoder"))
from text import TextEncoder
import torch.nn.functional as F 

def l2_normalize(x):
    return x/torch.norm(x,2,dim=1,keepdim=True)

class CLIP(nn.Module):
    #https://arxiv.org/pdf/2103.00020
    def __init__(self,
    Nb,#nb images texts pairs
    #VIT args
     H,W,C,D,N,dmodel,n_heads,num_classes,patch_resolution,
    #text embeder args
    nb_tokens_per_doc,VOCAB_SIZE,embed_in,embed_out,N_blocks,masked,
    #shared args 
    d_e,t
    ):
        super().__init__()
        self.I_f=ViT(H,W,C,D,N,dmodel,n_heads,num_classes,patch_resolution)#size N,D
        self.T_f=TextEncoder(nb_tokens_per_doc,VOCAB_SIZE,embed_in,dmodel,n_heads,embed_out,N_blocks,masked)#size N,embed_in
        self.W_i=nn.Linear(D,d_e,bias=False)
        self.W_t=nn.Linear(embed_in,d_e,bias=False)
        self.register_buffer("labels", torch.arange(Nb))
        self.t=nn.Parameter(torch.tensor(0.1))

    def forward(self,input_vit,input_text):

        I_f=self.I_f(input_vit)
        T_f=self.T_f(input_text)

        # calculate cos similarity
        I_e=l2_normalize(self.W_i(I_f))#N,de
        T_e=l2_normalize(self.W_t(T_f))#N,de
        logits=I_e@T_e.T * torch.exp(self.t) #N,N matrix 
        return logits 
    

class ClipLoss(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self,logits,classes):
        loss_i=F.cross_entropy(logits,classes)
        loss_t=F.cross_entropy(logits.T,classes)
        return (loss_i+loss_t)/2


       
if __name__ == "__main__":
    # --- batch & shared ---
    Nb = 4
    d_e = 128
    t_init = 0.1

    # --- ViT args ---
    H, W, C = 224, 224, 3
    D = 200
    N = 6                # number of ViT blocks
    dmodel = 14
    n_heads = 7
    num_classes = 10     # unused (final_linear is commented out)
    patch_resolution = 16

    # --- TextEncoder args ---
    nb_tokens_per_doc = 50
    VOCAB_SIZE = 1000
    embed_in = 200
    embed_out = embed_in * 2
    N_blocks = 12
    masked = True

    # --- dummy data ---
    images = torch.randn(Nb, C, H, W)
    texts  = torch.randint(0, VOCAB_SIZE, (Nb, nb_tokens_per_doc))


    clip = CLIP(
        Nb=Nb,
        H=H, W=W, C=C, D=D, N=N, dmodel=dmodel, n_heads=n_heads,
        num_classes=num_classes, patch_resolution=patch_resolution,
        nb_tokens_per_doc=nb_tokens_per_doc, VOCAB_SIZE=VOCAB_SIZE,
        embed_in=embed_in, embed_out=embed_out,
        N_blocks=N_blocks, masked=masked,
        d_e=d_e, t=t_init,
    )


    logits = clip(images, texts)
    print(f"logits shape: {logits.shape}")# expected: [Nb, Nb] = [4, 4]

    loss_fn = ClipLoss()
    classes = torch.arange(Nb)
    loss = loss_fn(logits, classes)
    print(f"loss: {loss.item():.4f}")

    loss.backward()
    print("backward OK ✓")




  
       



