import torch
import torch.nn as nn 


class TransformerBlock(nn.Module):
    def __init__(self,embed,dmodel,n_heads):
        super().__init__()
        """
        input is B,T,embed tensor 
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
        sft=torch.softmax(qk,dim=-2)
        prod=torch.einsum("bijn,bjkn->bikn",sft,v)#B,T,self.size,n_heads
        prod=prod.reshape(*prod.shape[0:2],-1)#B,T,dmodel
        return self.linear(prod)


if __name__=="__main__":
    x=torch.randn(10,20,30)

    m=TransformerBlock(30,40,4)
    print(m(x).shape)



