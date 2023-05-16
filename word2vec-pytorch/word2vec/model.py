import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""

# 跳词模型
class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        # 输入的均为稀疏向量（非独热编码）以减少显存/内存消耗
        # nn.Embedding的一个可选参数为sprase，默认为False，若改为True则梯度变为稀疏矩阵
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        # 权重的初始化为均匀分布
        init.xavier_uniform_(self.u_embeddings.weight)
        init.xavier_uniform_(self.v_embeddings.weight)
        # 初始化权重值

    def forward(self, centers, contexts_negatives): 
        # centers: (batch_size, 1)
        # contexts_negatives: (batch_size, max_len)
        v = self.v_embeddings(centers)
        # v: (batch_size, 1, emb_dimension)
        u = self.u_embeddings(contexts_negatives)
        # u: (batch_size, max_len, emb_dimension)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        # pred: (batch_size, 1, max_len)
        return pred


    def save_embedding(self, id2word, file_name):
        embedding = self.v_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
