import os
import re
import pickle
import multiprocessing
from functools import partial
# from bidict import bidict
from collections import defaultdict    

def valueGetKey(dfdict, value):
    for k, v in dfdict.items():
        if value in v:
            return k
    return None



def merge_one(kvt, son_dict):
    key, value = kvt
    son_rep = valueGetKey(son_dict, key)
    if not son_rep:
        raise Exception(f'{key}未在{son_dict}中找到代表序列！')
    

def merge(father_dfdict, sondfdict):
    pool = multiprocessing.Pool(processes=60)
    pool.map()
    retdfdict = defaultdict(list)
    for rep, pros in father_dfdict.items():
        son_rep = valueGetKey(sondfdict, rep)
        if not son_rep:
            raise Exception(f'{rep}未在{sondfdict}中找到代表序列！')
        retdfdict[son_rep] = list(set(pros) | set(sondfdict[son_rep]))
    return retdfdict


def getCdhitDict(path):
    clusters = defaultdict(list)
    with open(path, "r")as f:
        text = f.readlines()
        text = ''.join(text)
        pattern = re.compile(r">Cluster")
        cluster_start = [substr.start() for substr in re.finditer(pattern, text)] + [len(text)]
        
        for i in range(len(cluster_start)-1):
            block = text[cluster_start[i]:cluster_start[i+1]-1]
    #         print(block)
            block_lines = block.split('\n')
            mems = set()
            rep = None
            for line in block_lines:
                if "aa" in line:
                    startN = line.find(">")
                    endN = line.find("..")
                    # 没有点返回-1
                    protein = line[startN+1:endN]
                    if " " in protein:
                        print("切取错误！")
                        break
                    mems.add(protein)
                    if not rep and "*" in line:
                        rep = protein
                        print(line)
            if not rep:
                print("没有代表")
                print(block_lines)
                break
            
            for mem in mems:
                clusters[rep].append(mem)
    return clusters
    

# mmseqs2的结果

clusters_mmseqs2 = defaultdict(list)
with open("/home/yaoshuai/tools/MMseqs2/easy_resPros3_cdhit_cluster.tsv", "r") as f:
    text = f.readlines()
    for line in text:
        represent, member = line.split()
        clusters_mmseqs2[represent].append(member)
print("获取mmseqs聚类结果成功！")

clusters_cdhit3 = getCdhitDict("/home/yaoshuai/tools/cdhit/output/resPros3_cdhit.clstr")
print("获取第三次cdhit结果成功！")
            
mmseqsMergeCdhit = merge(clusters_cdhit3, clusters_mmseqs2)
print("合并mmseqs和第三次cdhit结果成功！")

clusters_cdhit2 = getCdhitDict("/home/yaoshuai/tools/cdhit/output/resPros2_cdhit.clstr")
print("获取第二次cdhit结果成功！")
clusters_cdhit1 = getCdhitDict("/home/yaoshuai/tools/cdhit/output/resPros1_cdhit.clstr")
print("获取第一次cdhit结果成功！")
cdhit2Merge3 = merge(mmseqsMergeCdhit, clusters_cdhit2)
print("合并mmseqs和第二次cdhit结果成功！")
cdhit1Merge3 = merge(mmseqsMergeCdhit, clusters_cdhit1)
print("合并mmseqs和第一次cdhit结果成功！")
with open("/home/yaoshuai/tools/merge/mergeGenome.pkl", 'w') as fp:
    pickle.dump(cdhit1Merge3, fp)
print("写入结果成功！")
