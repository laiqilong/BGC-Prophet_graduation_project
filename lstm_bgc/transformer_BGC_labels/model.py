import torch
import math
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable

from batch3Linear import batch3Linear
# from torch.nn.module import Transformer

# 
class transformerEncoderLabelsNet(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, num_encoder_layers: int=6, max_len: int=128, 
                 dim_feedforward: int=2048, dropout: float=0.1, labels_num: int=8, 
                 activation: Union[str, Callable[[Tensor], Tensor]]=F.gelu,
                #  custom_encoder: Optional[Any]=None,
                 layer_norm_eps: float=1e-05, batch_first: bool=True, norm_first: bool=True, concat_type: str='avg',
                 device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device':device, 'dtype':dtype}
        self.pos_encoder = PositionalEncoding(num_hiddens=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                   dropout, activation, layer_norm_eps, batch_first, norm_first,
                                                   **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.LabelsSqu = nn.Sequential(
        #     nn.LayerNorm(d_model, eps=layer_norm_eps),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1), 
        #     nn.ReLU()
        # )
        # self.LabelsOut = nn.Sequential(
        #     nn.Linear(max_len, labels_num),
        #     nn.Sigmoid()
        # )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_model, out_features=64), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=64, out_features=labels_num),
            nn.Sigmoid()
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.concat_type = concat_type

    def forward(self, src: Tensor, src_key_padding_mask: Tensor):
        # src: (batch_size, max_len, embed_dim)
        # src = src * math.sqrt(self.d_model)
        # 将mask转为布尔值，根据T/F选择性mask
        src_key_padding_mask = src_key_padding_mask.bool()
        # src_key_padding_mask = None
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        # memory: (batch_size, max_len, d_model)
        # mask的部分通过缩放点积注意力的softmax函数变为0，即此部分的注意力归0
        # labels = self.LabelsSqu(memory)
        # labels = self.LabelsOut(labels.squeeze())
        if self.concat_type == 'sum':
            memory = torch.sum(memory, dim=1)
        elif self.concat_type == 'avg':
            memory = torch.sum(memory, dim=1) / memory.shape[1]
        else:
            memory = memory[:, -1, :]
        labels = self.classifier(memory)
        # memory_reshaped = memory.reshape(memory.shape[1], memory.shape[0], -1)
        # TD = self.TDOut(memory_reshaped)
        # TD = TD.reshape(memory.shape[0], memory.shape[1], -1).squeeze(-1)
        return labels
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# class FeatLayer(nn.Module):
#     def __init__(self, max_len: int, dim_in: int, dim_out: int) -> None:
#         super().__init__()
#         sizes = []
#         while dim_in>dim_out:
#             sizes.append(dim_in)
#             dim_in //=4
#         self.feat = nn.Sequential(
#             nn.LayerNorm(sizes[0]),
#             *[nn.Sequential(batch3Linear(batch_size=max_len, in_features=sizes[i], 
#                                          out_features=sizes[i+1]), nn.Tanh()) for i in range(len(sizes)-1)],
#             batch3Linear(batch_size=max_len, in_features=sizes[-1], out_features=dim_out),
#             nn.Sigmoid()                                
#         )
        
#     def forward(self, inputs):
#         return self.feat(inputs)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens:int, dropout:float=0.1, max_len:int=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)