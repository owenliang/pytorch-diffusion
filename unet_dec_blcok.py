import torch
from torch import nn 
from unet_enc_block import EncoderBlock
from dataset import train_dataset
from time_position_emb import TimePositionEmbedding
from config import *

class DecoderBlock(nn.Module):
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

        # 反卷相当于在输入相邻像素间插入stride-1个0元素，尺寸公式是：stride*(input-1)+2*pad-kernel+2
        self.deconv=nn.ConvTranspose2d(out_channel,out_channel,kernel_size=2,stride=2,padding=0,dilation=1) # 不改通道数,尺寸翻倍
    
    def forward(self,x,t_emb):
        x=self.seq1(x)   # 改通道数,不改大小
        t_emb=self.relu(self.time_emb_linear(t_emb)).view(x.size(0),x.size(1),1,1)   # t_emb: (batch_size,out_channel,1,1) 
        x=self.seq2(x+t_emb)      # 不改通道数,不改大小
        return self.deconv(x)   # 反卷相当于在输入相邻像素间插入stride-1个0元素，尺寸公式是：stride*(input-1)+2*pad-kernel+2

if __name__=='__main__':
    # 图像x_t
    img=train_dataset[0][0]
    batch_x=img.unsqueeze(0).to(DEVICE)  

    # 时间步time
    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)  

    # time转embedding
    time_emb_size=32
    time_emb=TimePositionEmbedding(time_emb_size).to(DEVICE)
    batch_t_emb=time_emb(batch_t)

    # 编码
    encoder_block=EncoderBlock(1,64,time_emb_size).to(DEVICE)
    batch_encoder_z=encoder_block(batch_x,batch_t_emb)

    # 解码
    decoder_block=DecoderBlock(64,32,time_emb_size).to(DEVICE)
    batch_decoder_z=decoder_block(batch_encoder_z,batch_t_emb)

    print('batch_x:',batch_x.size())
    print('batch_encoder_z:',batch_encoder_z.size())
    print('batch_decoder_z:',batch_decoder_z.size())