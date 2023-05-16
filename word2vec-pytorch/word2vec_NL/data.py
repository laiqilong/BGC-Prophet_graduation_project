import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing
# from functools import partial
import itertools

np.random.seed(12345)

def getN(contexts, i, sampleWeights, k):
    generator = RandomGenerator(samplingWeights=sampleWeights)
    negatives = []
    print(f"Length of this contexts: {len(contexts)}")
    for context in tqdm(contexts, desc=f'Get negatives subprocess {i}', leave=True):
        negative = []
        while(len(negative)<len(context)*k):
            neg = generator.draw()
            if neg not in context:
                negative.append(neg)
        negatives.append(negative)
    return negatives, i


class RandomGenerator:

    def __init__(self, samplingWeights) -> None:
        self.population = list(range(0, len(samplingWeights)))
        self.samplingWeights = samplingWeights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = np.random.choice(self.population, DataReader.NEGATIVE_TABLE_SIZE, p=self.samplingWeights)
            self.i = 0
        self.i+=1
        return self.candidates[self.i-1]

class DataReader:
    # 负采样集合的大小
    NEGATIVE_TABLE_SIZE = int(1e8)

    def __init__(self, inputFileName, min_count, window_size, negativeNum):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.corpus = []

        self.inputFileName = inputFileName
        self.min_count = min_count
        self.window_size = window_size
        self.negativeNum = negativeNum

        self.centers = []
        self.contexts = []
        self.negatives = []

        self.__read_words(min_count) # 根据最小计数
        # self.__initTableNegatives()
        self.__initTableDiscards()
        self.__subsample() # 生成下采样后的语料库
        self.__getCentersAndContexts() # 生成中心词和周围词
        self.__getNegatives(k=self.negativeNum) # 生成负样本
        # 函数的调用等同于初始化

    def __read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1 #句子计数
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1 #单词计数
                        word_frequency[word] = word_frequency.get(word, 0) + 1 # 词频计算，用于构建词袋模型

                        if self.token_count % 1000000 == 0: #每一百万个输出一次
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            # 低于最小阈值的词不计入考量
            self.word2id[w] = wid # 建立单词到索引的字典
            self.id2word[wid] = w # 建立索引到单词的字典
            self.word_frequency[wid] = c # 词频字典
            wid += 1

        for line in open(self.inputFileName, encoding="utf-8"):
            line = line.split()
            if len(line) > 1:
                self.corpus.append([self.word2id[word] for word in line if word in self.word2id])
        print("generate corpus done!")
                # 生成原始词库，并去除低频词
        print("Total embeddings: " + str(len(self.word2id)))
        # 生成词袋模型

    def __initTableDiscards(self):
        # 抛弃一部分的词，即下采样
        # 词频超过t的词有概率被舍弃，所占比例越高越可能被抛弃
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / sum(self.word_frequency.values()) # 词频率的np.array
        # 
        # self.discards = np.sqrt(t / f) + (t / f) # np.array 丢弃的概率？
        # 更换丢弃概率的写法，此处f<t 仍有丢弃的风险
        self.discards = np.sqrt(t/f)
        print(self.discards.shape)
        # print(self.discards.shape)
        # 广播机制，使self.discards也为一个np.ndarrays

    def __subsample(self):
        def keep(w):
            # print(np.random.uniform(0, 1))
            # print(self.word2id[w])
            # print(self.discards[self.word2id[w]])
            return(np.random.uniform(0, 1) < self.discards[w])
        # print(type(self.corpus))
        print(max(self.word2id.values()))
        print(max(self.word_frequency.keys()))
        for i in tqdm(range(len(self.corpus)), desc='Subsample', leave=True):
            self.corpus[i] = [w for w in self.corpus[i] if keep(w)]


    def __initTableNegatives(self):
        # 采集负样本，计算采样词的采样概率，根据论文其幂应该为0.75
        # 此处为0.5，不清楚设置的含义，可能是为了提高低频词作为负样本的采样频率？
        # 按照d2l更改为0.75
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        # 采样频率为相对频率，因此需要累加，得到整数而非np.array
        ratio = pow_frequency / words_pow
        # 使用numpy计算绝对频率（累加频率，得到np.array而非单个数
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)  # count也为np.array类型
        # 负采样的频数设置为1e8，设置每个词取样的个数？
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c) # [wid]重复c次并合并list
        self.negatives = np.array(self.negatives) # 转成np.array类型
        np.random.shuffle(self.negatives) # 随机洗牌

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size] # 根据size依次对self.negatives进行拆分
        self.negpos = (self.negpos + size) % len(self.negatives) # 迭代步骤，每次都增加size
        if len(response) != size: # 取不满size个时
            return np.concatenate((response, self.negatives[0:self.negpos])) #从头拼凑以符合要求
        return response
        # 在函数getNegatives中，中心词的周围词也可能被渠道，因此不准确

    # 完成DataReader定义，此处的DataReader并没有实际包装出中心词、周围词和负样本与padding，因此不够完善

    def __getCentersAndContexts(self):
        for line in tqdm(self.corpus, desc='getCentersAndContexts', leave=True):
            if len(line) < 2:
                continue
            self.centers += line
            for i in range(len(line)):
                boundary = np.random.randint(1, self.window_size)
                indices = list(range(max(i-boundary, 0), min(len(line), i+1+boundary)))
                indices.remove(i)
                self.contexts.append([line[idx] for idx in indices])
        # 完成中心词和上下文的生成
    
    def __getNegatives(self, k=3):
        sampleWeights = np.array(list(self.word_frequency.values())) ** 0.75
        sampleWeights /= sampleWeights.sum()
        # generator = RandomGenerator(samplingWeights=sampleWeights)
        # def getN(contexts, i):
        #     generator = RandomGenerator(samplingWeights=sampleWeights)
        #     negatives = []
        #     print(f"Length of this contexts: {len(contexts)}")
        #     for context in tqdm(contexts, desc=f'Get negatives subprocess {i}', leave=True):
        #         negative = []
        #         while(len(negative)<len(context)*k):
        #             neg = generator.draw()
        #             if neg not in context:
        #                 negative.append(neg)
        #         negatives.append(negative)
        #     return (negatives, i)

        pool = multiprocessing.Pool(processes=40)
        contextsLen = len(self.contexts)
        jobs = []
        for i in range(40):
            interval = self.contexts[int(i*contextsLen/40):int((i+1)*contextsLen/40)]
            jobs.append(pool.apply_async(getN, args=(interval, i, sampleWeights, k)))
        pool.close()
        pool.join()
        results = []
        for job in jobs:
            results.append(job.get())
        results.sort(key=lambda a:a[1])
        for res in results:
            self.negatives += res[0]

        # for context in tqdm(self.contexts, desc='get Negatives', leave=True):
        #     negative = []
        #     while(len(negative)<len(context)*k):
        #         neg = generator.draw()
        #         if neg not in context:
        #             negative.append(neg)
        #     self.negatives.append(negative)
        # print("Generate negatives done!")





# -----------------------------------------------------------------------------------------------------------------

# 根据DataReader类生成word2vec数据集
class Word2vecDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print(len(self.data.centers))
        print(len(self.data.contexts))
        print(len(self.data.negatives))
        assert len(self.data.centers) == len(self.data.contexts) == len(self.data.negatives)
        # self.input_file = open(data.inputFileName, encoding="utf8")
        # 该程序每段open都没有对应的close，习惯不是很好

    def __len__(self):
        return len(self.data.centers)
        # 调用属性的方法/属性
        # 肯定会出现一个句子只有一个单词的情况，此时直接从头开始读

    def __getitem__(self, idx):
        return (self.data.centers[idx], self.data.contexts[idx], self.data.negatives[idx])
        
    
    # 静态方法，可以由类名调用 
    @staticmethod
    def collate(batches):
        max_len = max(len(context)+len(negative) for _, context, negative in batches)
        # 所有batch中最长的序列作为定长，然后padding
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in batches:
            cur_len = len(context)+len(negative)
            centers += [center]
            contexts_negatives += [context + negative + [0]*(max_len-cur_len)]
            masks += [[1]*cur_len + [0]*(max_len-cur_len)]
            labels += [[1]*len(context) + [0]*(max_len-len(context))]
        return (torch.tensor(centers).reshape((-1, 1)),
                torch.tensor(contexts_negatives),
                torch.tensor(masks),
                torch.tensor(labels))
