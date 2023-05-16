from collections import defaultdict
import re
import pickle
def procCdhitRes(path, clusters, rep_clusters):
    # clusters_cdhit3 = {}
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
                    BGCs = line[startN+1:endN]
                    if " " in BGCs:
                        print("切取错误！")
                        break
                    mems.add(BGCs)
                    if not rep and "*" in line:
                        rep = BGCs
                        # print(line)
            if not rep:
                print("没有代表")
                print(block_lines)
                break
            mmseqs_rep = rep_clusters[rep]
            for mem in mems:
                rep_clusters[mem] = mmseqs_rep
            clusters[mmseqs_rep]  = clusters[mmseqs_rep] | mems
            # print(mems)

with open("/home/yaoshuai/tools/MMseqs2/easy_res_norep_cluster.tsv", "r") as f:
    text = f.readlines()
    clusters = defaultdict(set)
    rep_clusters = {}
    for line in text:
        represent, member = line.split()
#         print(represent, member)
        clusters[represent].add(member)
        rep_clusters[member] = represent
    print("处理mmseqs2结果完成！")
    
procCdhitRes("/home/yaoshuai/tools/cdhit/output/resProteins_norep_cdhit.clstr", clusters, rep_clusters)
print("合并cdhit结果完成！")

with open("/home/yaoshuai/tools/merge/mergeNorepCluster.pkl", 'wb') as fp:
    pickle.dump(clusters, fp)
print("写入clusters结果成功！")
with open("/home/yaoshuai/tools/merge/mergeNorepClusterRep.pkl", 'wb') as fp:
    pickle.dump(rep_clusters, fp)
print("写入clusters结果成功！")

clusters_id = defaultdict(list)
rep2id = {}
pro2rep_id = {}
count = 1
for repPro, clusterPros in clusters.items():
    id = 'Cluster_{:0>7}'.format(count)
    rep2id[repPro] = id
    clusters_id[id] = clusterPros
    count+=1
print("生成clusters转换关系成功！")
for pro, repPro in rep_clusters.items():
    pro2rep_id[pro] = rep2id[repPro]
print("生成rep_clusters转换关系成功！")

with open("/home/yaoshuai/tools/merge/clusters_id.pkl", 'wb') as fp:
    pickle.dump(clusters_id, fp)
print("保存clusters_id变量成功！")

with open("/home/yaoshuai/tools/merge/rep2id.pkl", 'wb') as fp:
    pickle.dump(rep2id, fp)
print("保存rep2id变量成功！")

with open("/home/yaoshuai/tools/merge/pro2rep_id.pkl", 'wb') as fp:
    pickle.dump(pro2rep_id, fp)
print("保存pro2rep_id变量成功！")

