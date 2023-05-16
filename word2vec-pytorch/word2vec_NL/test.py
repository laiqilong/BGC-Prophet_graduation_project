import torch
from data import DataReader
from model import SkipGramModel

class testWord2vec:
    def __init__(self, dataPath, modelPath) -> None:
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.data = DataReader(dataPath, min_count=5, window_size=5, negativeNum=5)
        self.model = SkipGramModel(emb_size=len(self.data.word2id), emb_dimension=200)
        self.model.load_state_dict(torch.load(modelPath))
        # self.embed = self.model.v_embeddings

def get_similar_tokens(query_token, k, test):

    W = test.model.u_embeddings.weight.data
    x = W[test.data.word2id[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
    torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]: # 删除输⼊词
        print(f'cosine sim={float(cos[i]):.3f}: {test.data.id2word[i]}')
get_similar_tokens('chip', 20, testWord2vec('ptb.train.txt', 'output_ptb_10.pt'))

