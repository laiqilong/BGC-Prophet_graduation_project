import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

from data import DataReader, Word2vecDataset
from model import SkipGramModel
from loss import SimoidBCELoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 使用双卡并行

class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=200, batch_size=4096, window_size=5, iterations=30,
                 initial_lr=0.002, min_count=5):

        self.data = DataReader(input_file, min_count=min_count, window_size=window_size, negativeNum=5, mode='load_and_train')
        # 读取文件，为耗时步骤
        dataset = Word2vecDataset(self.data)
        # 将DataReader包装一层，用于生成输入、输出词向量
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=10, collate_fn=dataset.collate)
        # collate_fn 用于生成批次，每个数据分开成单一批次

        self.output_file_name = output_file
        # 嵌入的独热向量尺寸和词个数一致
        self.emb_size = len(self.data.word2id)
        # 嵌入向量的维度
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = nn.DataParallel(SkipGramModel(self.emb_size, self.emb_dimension))
        self.lossFunction = SimoidBCELoss()
        # 设置多卡并行

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # self.device = "cpu"
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        # 训练过程
        optimizer = optim.Adam(self.skip_gram_model.parameters(), lr=self.initial_lr)
        print(self.skip_gram_model)
        # 可变学习率为cosine退火
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6, eta_min=0.0001, verbose=True)

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            # 优化器为稀疏的亚当优化器
            # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            # # 可变学习率为cosine退火
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            # 根据batch拆分数据
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                # 转为torch的tensor类型，转移到对应设备
                centers = sample_batched[0].to(self.device)
                contexts_negatives = sample_batched[1].to(self.device)
                masks = sample_batched[2].to(self.device)
                labels = sample_batched[3].to(self.device)
                # print("centers:", centers)
                # print("centers.shape:", centers.shape)
                # print("contexts_negatives.shape", contexts_negatives.shape)
                # print("torch.max(centers):", torch.max(centers))
                # print("torch.max(contexts_negatives)", torch.max(contexts_negatives))
                
                optimizer.zero_grad()
                # 梯度归零
                # loss 包含在模型里面，因此模型直接返回的就是loss
                pred = self.skip_gram_model(centers, contexts_negatives)
                # pred: (batch_size, 1, max_len)
                # labels: (batch_size, max_len)
                # masks: (batch_size, max_len)
                loss = (self.lossFunction(pred.reshape(labels.shape).float(), labels.float(), masks) /masks.sum(axis=1)*masks.shape[1])
                loss.sum().backward()
                optimizer.step()
                # 每一次都进行梯度归零和梯度反向传播的操作
                
                # 更改为每一百次记录一次loss
                if i > 0 and i % 400 == 0:
                    print(" Loss: " + str((loss.sum()/loss.numel()).data))
            
            scheduler.step()
            # print(f"Epoch {iteration} learning rate:{scheduler.get_lr()}")

            # self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name+'_'+str(iteration)+'.vec')
            path = self.output_file_name+'_'+str(iteration)+'.pt'
            # try:
            #     torch.save(self.skip_gram_model.state_dict(), path)
            #     print("use model.state_dict method to save model")
            # except:
            torch.save(self.skip_gram_model.module.state_dict(), path)
            print("use model.module.state_dict method to save model")


if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file=r"/home/yaoshuai/data/corpus/corpus_mibig.txt", output_file="output_d2l_2gpu_subsample")
    w2v.train()
