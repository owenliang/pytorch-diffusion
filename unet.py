import torch 
from torch import nn 
from dataset import train_dataset
from unet_enc_block import EncoderBlock
from unet_dec_blcok import DecoderBlock
from config import * 
from diffusion import forward_diffusion

class UNet(nn.Module):
    def __init__(self,img_channel,channels=[64, 128, 256, 512, 1024]):
        super().__init__()

        channels=[img_channel]+channels

        self.enc_blocks=nn.ModuleList()
        for i in range(len(channels)-1):
            self.enc_blocks.append(EncoderBlock(channels[i],channels[i+1]))
        
        self.dec_blocks=nn.ModuleList()
        for i in range(len(channels)-1):
            self.dec_blocks.append(DecoderBlock(channels[-i-1]*2,channels[-i-2])) # 有残差结构,所以channal输入翻倍
        
    def forward(self,x,t):
        # TODO: t embedding
        residual=[]
        for enc_block in self.enc_blocks:
            x=enc_block(x)
            residual.append(x)
        for dec_block in self.dec_blocks:
            residual_x=residual.pop(-1)
            x=dec_block(torch.cat((residual_x,x),dim=1))
        return x 
        

if __name__=='__main__':
    batch_x=torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE) # 2个图片拼batch, (2,1,96,96)
    batch_x=batch_x*2-1 # 像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)  # 每张图片随机生成diffusion步数
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)

    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    unet=UNet(batch_x_t.size(1))
    batch_predict_noise_t=unet(batch_x_t,batch_t)
    print('batch_predict_noise_t:',batch_predict_noise_t.size())