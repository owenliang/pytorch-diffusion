import torch 
from torch import nn 
from config import * 
import math 

class CrossAttention(nn.Module):
    def __init__(self,channel,qsize,vsize,fsize,cls_emb_size):
        super().__init__()
        self.w_q=nn.Linear(channel,qsize)
        self.w_k=nn.Linear(cls_emb_size,qsize)
        self.w_v=nn.Linear(cls_emb_size,vsize)
        self.softmax=nn.Softmax(dim=-1)
        self.z_linear=nn.Linear(vsize,channel)
        self.norm1=nn.LayerNorm(channel)
        # feed-forward结构
        self.feedforward=nn.Sequential(
            nn.Linear(channel,fsize),
            nn.ReLU(),
            nn.Linear(fsize,channel)
        )
        self.norm2=nn.LayerNorm(channel)
    
    def forward(self,x,cls_emb): # x:(batch_size,channel,width,height), cls_emb:(batch_size,cls_emb_size)
        x=x.permute(0,2,3,1) # x:(batch_size,width,height,channel)
        
        # 像素是Query
        Q=self.w_q(x)   # Q: (batch_size,width,height,qsize)
        Q=Q.view(Q.size(0),Q.size(1)*Q.size(2),Q.size(3))   # Q: (batch_size,width*height,qsize)

        # 引导分类是Key和Value
        K=self.w_k(cls_emb) # K: (batch_size,qsize)
        K=K.view(K.size(0),K.size(1),1) # K: (batch_size,qsize,1)
        V=self.w_v(cls_emb) # V: (batch_size,vsize)
        V=V.view(V.size(0),1,V.size(1))  # v: (batch_size,1,vsize)

        # 注意力打分矩阵Q*K
        attn=torch.matmul(Q,K)/math.sqrt(Q.size(2)) # attn: (batch_size,width*height,1)
        attn=self.softmax(attn) # attn: (batch_size,width*height,1)
        # print(attn) # 就一个Key&value，所以Query对其注意力打分总是1分满分

        # 注意力层的输出
        Z=torch.matmul(attn,V)    # Z: (batch_size,width*height,vsize)
        Z=self.z_linear(Z)  # Z: (batch_size,width*height,channel)
        Z=Z.view(x.size(0),x.size(1),x.size(2),x.size(3))   # Z: (batch_size,width,height,channel)

        # 残差&layerNorm
        Z=self.norm1(Z+x)# Z: (batch_size,width,height,channel)

        # FeedForward
        out=self.feedforward(Z)# Z: (batch_size,width,height,channel)
        # 残差&layerNorm
        out=self.norm2(out+Z)
        return out.permute(0,3,1,2)

if __name__=='__main__':
    batch_size=2
    channel=1
    qsize=256
    cls_emb_size=32
    
    cross_atn=CrossAttention(channel=1,qsize=256,vsize=128,fsize=512,cls_emb_size=32)
    
    x=torch.randn((batch_size,channel,IMG_SIZE,IMG_SIZE))
    cls_emb=torch.randn((batch_size,cls_emb_size)) # cls_emb_size=32

    Z=cross_atn(x,cls_emb)
    print(Z.size())     # Z: (2,1,48,48)