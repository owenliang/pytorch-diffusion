from torch import nn 
from dataset import train_dataset

class EncoderBlock(nn.Module):
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

        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0) # 不改通道数,尺寸减半
    
    def forward(self,x):
        x=self.seq1(x)
        x=self.seq2(x)
        return self.maxpool(x)

if __name__=='__main__':
    img=train_dataset[0][0]
    batch_x=img.unsqueeze(0)

    encoder_block=EncoderBlock(1,64)
    batch_z=encoder_block(batch_x)
    print('batch_x:',batch_x.size())
    print('batch_z:',batch_z.size())