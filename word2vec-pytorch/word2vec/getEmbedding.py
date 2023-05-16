import pickle
import os
import numpy as np
from model import SkipGramModel
import torch
import sys

def getEmbedding(modelPath, emb_size, name, epoch, emb_dimension=100):
    model = SkipGramModel(emb_size=emb_size, emb_dimension=emb_dimension)
    model.load_state_dict(torch.load(modelPath))
    print(model)
    embed = model.v_embeddings.weight.cpu().data.numpy()
    print(embed.shape)
    np.save(f"embed_{name}_{epoch}.npy", embed)



if __name__ == '__main__':
    modelPath = sys.argv[1]
    name = sys.argv[2]
    emb_size = int(sys.argv[3])
    epoch = sys.argv[4]
    emb_dimension = int(sys.argv[5])
    print("modelPath:", modelPath)
    print("name:", name)
    print("emb_size:", emb_size)
    print("epoch:", epoch)
    print("emb_dimension:", emb_dimension)
    # for i in range(10):
    #     modelPath = f'./output_mibig_{i}.pt'
    #     name = 'mibig'
    getEmbedding(modelPath=modelPath, emb_size=emb_size, name=name, epoch=epoch, emb_dimension=emb_dimension)
    