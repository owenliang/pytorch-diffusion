from torch import nn 

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_emb_size):
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

    def forward(self,x,t_emb): # t_emb: (batch_size,time_emb_size)
        x=self.seq1(x)  # 改通道数,不改大小
        t_emb=self.relu(self.time_emb_linear(t_emb)).view(x.size(0),x.size(1),1,1)   # t_emb: (batch_size,out_channel,1,1) 
        return self.seq2(x+t_emb)        # 不改通道数,不改大小
