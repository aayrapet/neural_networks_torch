import torch
import torch.nn as nn 


class TransformerBlock(nn.Module):
    def __init__(self,embed,dmodel,n_heads,T,masked=False):
        super().__init__()
        """
        input is B,T,embed tensor 

        model.tocuda() moves all nn;parameters to cuda, but not those define as torch.ones() torch .tril so register uffer them so they are also automatically paqssed to cuda there!!


        """
        self.Q=nn.Linear(embed,dmodel,bias=False)
        self.K=nn.Linear(embed,dmodel,bias=False)
        self.V=nn.Linear(embed,dmodel,bias=False)
        if dmodel% n_heads !=0:
            raise ValueError('dmodel has to be divisible by n_heads ')

        self.size=int(dmodel/n_heads)
        self.scale=self.size**-0.5
        self.n_heads=n_heads
        self.linear=nn.Linear(dmodel,embed)
        self.masked=masked
        if self.masked:  
            #let upper triangle be 0 , lower 1
            self.register_buffer("tril", torch.tril(torch.ones(T,T)))

    def forward(self, x):
        q=self.Q(x)# (B,T,dmodel)
        k=self.K(x)
        v=self.V(x)
        
        return self.attention(q,k,v)

    def attention(self,q,k,v):

        q=q.reshape(*q.shape[0:2],self.size,self.n_heads)
        k=k.reshape(*q.shape[0:2],self.size,self.n_heads)
        v=v.reshape(*q.shape[0:2],self.size,self.n_heads)
        #we get qi i=1....n_heads, each qi of size B,T,self.size
        qk=torch.einsum("bijn,bkjn->bikn",q,k)*self.scale

        if self.masked:
        #mask attention 
            B,T,_,_=qk.shape
            #just reshape for dim match 
            tril=self.tril[:,:T,:T,:].unsqueeze(0).unsqueeze(-1).expand(B,-1,-1,self.n_heads)
            #0s are now -inf since exp(-inf)=0
            #effective T can be lower then declared T at init (at inference)
            qk=qk.masked_fill(tril==0,-float("inf"))

        sft=torch.softmax(qk,dim=-2)
        prod=torch.einsum("bijn,bjkn->bikn",sft,v)#B,T,self.size,n_heads
        prod=prod.reshape(*prod.shape[0:2],-1)#B,T,dmodel
        return self.linear(prod)


if __name__=="__main__":
    x=torch.randn(10,20,30)

    m=TransformerBlock(30,40,4,20)
    print(m(x).shape)



