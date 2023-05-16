import torch
from torch import nn
from batch3Linear import batch3Linear
from ATTENTION import *

# class selfMA(nn.Module):
#     def __init__(self, embed_dim:int, num_heads:int, batch_first:bool=True) -> None:
#         super().__init__()
#         self.MA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)

#     def forward(self, x):
#         x, _ = self.MA(x, x, x)
#         return x

class LSTMTimeDistributedNet(nn.Module):
    def __init__(self, embedding_dim:int, num_heads:int, depth:int,
                 hidden_dim:int, num_layers:int, max_len:int, 
                 labels_num:int, dropout:int=0.5, initWeight:bool=False) -> None:
        super().__init__()

        # self.MAlayers = [selfMA(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True) for _ in range(depth)]
        # self.depth = depth
        # self.MAblock = nn.Sequential(*self.MAlayers)
        # self.attention = ATTBlock(dim_input=embedding_dim, depth=depth)

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        # 全连接层的权重相互独立，用于预测每个基因是否是BGC的组成部分
        self.TD = TimeDistributed(FeatLayer(max_len, hidden_dim*2, 1))
        # self.TD = nn.Sequential(
        #     nn.LayerNorm(hidden_dim*2),
        #     nn.Dropout(dropout),
        #     nn.Linear(in_features=hidden_dim*2, out_features=1),
        #     # nn.GELU(),
        #     # nn.Linear(in_features=128, out_features=32),
        #     # nn.GELU(),
        #     # nn.Linear(in_features=32, out_features=1),
        #     nn.Sigmoid()
        # )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim*2, out_features=64), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=64, out_features=labels_num),
            nn.Sigmoid()
        )
        if initWeight:
            self.__initWeight()

    def forward(self, inputs):
        # x = self.MAblock(inputs)
        # x = self.attention(inputs)
        # print(x.shape)
        x, _ = self.LSTM(inputs, None)
        TD = self.TD(x)
        memory = torch.sum(x, dim=1) / x.shape[1]
        labels = self.classifier(memory)
        # 输出是否是BGC的组成部分，以及BGC种类两个信息
        return labels, TD.squeeze()
    
    def __initWeight(self):
        pass


class FeatLayer(nn.Module):
    # (batch_size, max_len, 2*hidden_dim)
    # 上述结构不对，应该改为(batch_size, max_len*hidden_dim *2)
    # 上述结构也不对，使用bmm实现新形式的batch3Linear
    # input: (batch_size, max_len, hidden_dim*2) -> (max_len, batch_size, hidden_dim*2)
    def __init__(self, max_len:int, dim_in:int, dim_out:int) -> None:
        super().__init__()
        sizes = []
        # 需要保证每个时间步都至少有一个输出
        # 如输入为(batch_size, max_len, hidden_dim*2)->(max_len, batch_size, hidden_dim*2)
        # hidden_dim*2=256*2
        # 则sizes: [256*2, 64, 8]
        while dim_in> dim_out:
            sizes.append(dim_in)
            dim_in //=8
            # 多次降采样

        # self.feat = nn.Sequential(
        #     *[nn.Linear(dim_i, dim_i // 8) for dim_i in sizes[:-1]],
        #     nn.Linear(sizes[-1], dim_out),
        #     nn.Sigmoid()
        # )

        # batch3Linear weight:(batch_size, in_features, out_features), input(max_len, batch_size, in_features)
        # bmm :torch.bmm(input, self.weight) + self.bias
        # output: (max_len, batch_size, out_features)

        self.feat = nn.Sequential(
            nn.LayerNorm(sizes[0]),
            *[nn.Sequential(batch3Linear(batch_size=max_len, in_features=sizes[i], out_features=sizes[i+1]), nn.GELU()) for i in range(len(sizes)-1)],
            batch3Linear(batch_size=max_len, in_features=sizes[-1], out_features=dim_out), #(batch_size, max_len, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch_size*max_len, 1)
        # 上述输出结构不正确， 应该为(batch_size, dim_out)
        # 此时dim_out 为max_len大小
        return self.feat(x)


class TimeDistributed(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        if len(x.shape) <= 2:
            return self.module(x)
        # 将输入数据reshape为 (batch_size * max_len, 2*embed_dim)
        # 上述reshape方式不正确，应该保留batch_size 维度
        # 变为(batch_size, max_len*embed_dim)
        # 上述reshape方式也不对，应保留max_len和batch_size，自行实现batch linear
        # 输入(max_len, batch_size, hidden_dim*2)
        x_reshaped = x.reshape(x.shape[1], x.shape[0], -1)
        y = self.module(x_reshaped)
        # 输出(max_len, batch_size, 1)
        # 还原原始形状
        y = y.reshape(x.shape[0], x.shape[1], -1)
        y = y.squeeze(-1)
        # (batch_size, max_len)
        return y
    
    



