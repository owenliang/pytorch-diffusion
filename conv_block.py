from torch import nn 
from cross_attn import CrossAttention

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_emb_size,qsize,vsize,fsize,cls_emb_size):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1), # 改通道数,不改大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self.time_emb_linear=nn.Linear(time_emb_size,out_channel)    # Time时刻emb转成channel宽,加到每个像素点上
        self.relu=nn.ReLU()

        self.seq2=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1), # 不改通道数,不改大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        # 像素做Query，计算对分类ID的注意力，实现分类信息融入图像，不改变图像形状和通道数
        self.crossattn=CrossAttention(channel=out_channel,qsize=qsize,vsize=vsize,fsize=fsize,cls_emb_size=cls_emb_size)

    def forward(self,x,t_emb,cls_emb): # t_emb: (batch_size,time_emb_size)
        x=self.seq1(x)  # 改通道数,不改大小
        t_emb=self.relu(self.time_emb_linear(t_emb)).view(x.size(0),x.size(1),1,1)   # t_emb: (batch_size,out_channel,1,1) 
        output=self.seq2(x+t_emb)        # 不改通道数,不改大小
        return self.crossattn(output,cls_emb)   # 图像和引导向量做attention