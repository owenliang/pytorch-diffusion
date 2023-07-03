import torch 
from torch import nn 
from dataset import train_dataset
from config import * 
from time_position_emb import TimePositionEmbedding

class EncoderBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_emb_size):
        super().__init__()

        self.seq1=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1), # 改通道数,不改大小
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )

        self.time_emb_linear=nn.Linear(time_emb_size,out_channel)    # Time时刻emb转成channel宽,加到每个像素点上
        self.relu=nn.ReLU()

        self.seq2=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1), # 不改通道数,不改大小
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )

        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0) # 不改通道数,尺寸减半
    
    def forward(self,x,t_emb): # t_emb: (batch_size,time_emb_size)
        x=self.seq1(x)  # 改通道数,不改大小
        t_emb=self.relu(self.time_emb_linear(t_emb)).view(x.size(0),x.size(1),1,1)   # t_emb: (batch_size,out_channel,1,1) 
        x=self.seq2(x+t_emb)        # 不改通道数,不改大小
        return self.maxpool(x)  # 不改通道数,尺寸减半

if __name__=='__main__':
    # 图像x_t
    img=train_dataset[0][0]
    batch_x=img.unsqueeze(0)
    
    # 时间步time
    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)  
    
    # time转embedding
    time_emb_size=32
    time_emb=TimePositionEmbedding(time_emb_size)
    batch_t_emb=time_emb(batch_t)
    
    # 编码
    encoder_block=EncoderBlock(1,64,time_emb_size)
    batch_z=encoder_block(batch_x,batch_t_emb)
    print('batch_x:',batch_x.size())
    print('batch_z:',batch_z.size())