import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import pickle
import random

class sentence_sampling:
    def __init__(self, word_list, weights, sentences_length):
        self.word_list = word_list
        self.sentences_length = sentences_length
        self.weights = weights

    def generate_sentence(self, plot=False):
        sentences = []
        for sentence_length in self.sentences_length:
            sentence = ' '.join(random.choices(population=self.word_list, weights=self.weights, k=sentence_length))
            #print(sentence)
            sentences.append(sentence)
        if plot:
            # 绘制直方图
            plt.hist(sentences_length, bins=30, density=True, alpha=0.6, color='b')
            plt.title("Length of chi2 Sampling")
            plt.xlabel('Length')
            plt.ylabel('Frequency')
            plt.savefig('Sampling_chi2_2.png')
        return sentences

    def add_label(self, label: str, sentences: list):
        sentences_list = []
        for sentence in sentences:
            sentences_list.append(label+sentence)
        return sentences_list

length_file_path = '/home/yaoshuai/data/negsample/bgc_len.txt'
possentence_file_path = '/home/yaoshuai/data/corpus/corpus_mibig.txt'
train_label_path = '/home/yaoshuai/tools/BGC_labels_pred/data/training_label.txt'
possentences_path = '/home/yaoshuai/data/negsample/possentences.pkl'
possentences_path2 = '/home/yaoshuai/data/negsample/possentences2.pkl'
negsentences_path = '/home/yaoshuai/data/negsample/negsentences2.pkl'
word_frequency_dic_path = '/home/yaoshuai/data/negsample/word_frequency_dic_min3.pkl'
mibig_word_frequency_dic_path = '/home/yaoshuai/data/negsample/mibig_word_frequency_dic.pkl'
with open(length_file_path, 'r')as fl, open(possentences_path, 'rb')as fp, open(word_frequency_dic_path, 'rb')as fw:
    length_data = fl.read().split()[1::2]
    length_data_num = len(length_data)
    length_data = list(map(int, length_data))
    #取出mibig中的bgc序列
    lines = pickle.load(fp)
    sentences = [line.strip('\n')[10:] for line in lines]
    print(sentences[567:599])
    possentences = []
    possentences_word = []
    mibig_word_frequency_dic = {}
    #去掉长度为1的序列
    for sentence in sentences:
        sentence_word = sentence.split(' ')
        for word in sentence_word:
            mibig_word_frequency_dic[word] = mibig_word_frequency_dic.get(word, 0) + 1
        if len(sentence_word)<=1:
            continue
        else:
            possentences.append(sentence)
            possentences_word = possentences_word + sentence_word
    
    print(f'mibig词总数：{len(mibig_word_frequency_dic)}')

    print('去重前mibig中基因数量：', len(possentences_word))
    poseword = list(set(possentences_word))# 长度>1的bgc中的所有基因（id）
    poseword_num = len(poseword)
    print('去重后mibig中基因数量：', poseword_num)
    #取出genome中的所有min_count=3的基因（id）
    word_frequency_dic = pickle.load(fw)
    genome_word_list = list(word_frequency_dic.keys())
    '''#随机抽取
    genome_selected_word = list(np.random.choice(genome_word_list, poseword_num))
    #总负单词样本
    word_list = poseword + genome_selected_word'''

df = 2.7008291842129593  # 自由度参数
loc = 0.7619598528358154  # 位置参数
scale = 5.7855219372765525  # 缩放参数
#按照卡方分布采样
sentences_length = np.round(chi2.rvs(df, loc=loc, scale=scale, size=length_data_num*3)).astype(int)
print(f'negsentences_length:\n{sentences_length}')
#创建权重数组
mibig_word_counts = []

for word in poseword:
    if word in word_frequency_dic:
        mibig_word_counts.append(mibig_word_frequency_dic[word])
        word_frequency_dic[word] = mibig_word_frequency_dic[word]#防止会把min_count<3的重新添加进word_frequency_dic
print(f"min_count=3的总字典中mibig基因的数量：{len(mibig_word_counts)}")
print(f"min_count=3的总字典中基因的数量：{len(word_frequency_dic)}")
mibig_word_total_counts = sum(mibig_word_counts)
word_total_counts = sum(list(word_frequency_dic.values()))
for word in poseword:
    try:
        word_frequency_dic[word] = word_frequency_dic[word]*((word_total_counts - mibig_word_total_counts)/mibig_word_total_counts)
    except:
        print(f'word_frequency_dic 中没有 {word}')
        continue
print(f'字典中mibig基因总counts：{mibig_word_total_counts}')
print(f'字典中总counts：{word_total_counts}')
weights = list(word_frequency_dic.values())
#采样
negsampling = sentence_sampling(genome_word_list, weights, sentences_length)
negsentences = negsampling.generate_sentence(plot=True)
#sentences前加label
poslabel = '1 +++$+++ '
neglabel = '0 +++$+++ '
possentences = negsampling.add_label(poslabel, possentences)
negsentences = negsampling.add_label(neglabel, negsentences)
tatal_list = possentences + negsentences
#打乱顺序
random.shuffle(tatal_list)
#保存
with open(train_label_path, 'w+')as ft, open(possentences_path2, 'wb')as fp, open(negsentences_path, 'wb')as fn, open(mibig_word_frequency_dic_path, 'wb')as fm:
    for i in tatal_list:
        ft.write(i+'\n')
    pickle.dump(possentences, fp)
    pickle.dump(negsentences, fn)
    pickle.dump(mibig_word_frequency_dic, fm)
