import torch 
from torch import nn 
import math 
from config import *

class TimePositionEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_emb_size=emb_size//2
        half_emb=torch.exp(torch.arange(self.half_emb_size)*(-1*math.log(10000)/(self.half_emb_size-1)))
        self.register_buffer('half_emb',half_emb)

    def forward(self,t):
        t=t.view(t.size(0),1)
        half_emb=self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size)
        half_emb_t=half_emb*t
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
        return embs_t
    
if __name__=='__main__':
    time_pos_emb=TimePositionEmbedding(8)
    t=torch.randint(0,T,(2,))   # 随机2个图片的time时刻
    embs_t=time_pos_emb(t) 
    print(embs_t)