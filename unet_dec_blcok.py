from torch import nn 
from unet_enc_block import EncoderBlock
from dataset import train_dataset

class DecoderBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.seq1=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1), # 改通道数,不改大小
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )

        # TODO : T步时间embedding转成channel宽向量,加到每个像素点上

        self.seq2=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1), # 不改通道数,不改大小
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )

        # 反卷相当于在输入相邻像素间插入stride-1个0元素，尺寸公式是：stride*(input-1)+2*pad-kernel+2
        self.deconv=nn.ConvTranspose2d(out_channel,out_channel,kernel_size=2,stride=2,padding=0,dilation=1) # 不改通道数,尺寸翻倍
    
    def forward(self,x):
        x=self.seq1(x)
        x=self.seq2(x)
        return self.deconv(x)

if __name__=='__main__':
    img=train_dataset[0][0]
    batch_x=img.unsqueeze(0)

    encoder_block=EncoderBlock(1,64)
    batch_encoder_z=encoder_block(batch_x)

    decoder_block=DecoderBlock(64,32)
    batch_decoder_z=decoder_block(batch_encoder_z)

    print('batch_x:',batch_x.size())
    print('batch_encoder_z:',batch_encoder_z.size())
    print('batch_decoder_z:',batch_decoder_z.size())