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

with open("/home/yaoshuai/tools/MMseqs2/easy_resPros3_cdhit_cluster.tsv", "r") as f:
    text = f.readlines()
    clusters = defaultdict(set)
    rep_clusters = {}
    for line in text:
        represent, member = line.split()
#         print(represent, member)
        clusters[represent].add(member)
        rep_clusters[member] = represent
    print("处理mmseqs2结果完成！")
    
procCdhitRes("/home/yaoshuai/tools/cdhit/output/resPros3_cdhit.clstr", clusters, rep_clusters)
print("合并cdhit3结果完成！")
procCdhitRes("/home/yaoshuai/tools/cdhit/output/resPros2_cdhit.clstr", clusters, rep_clusters)
print("合并cdhit2结果完成！")
procCdhitRes("/home/yaoshuai/tools/cdhit/output/resPros1_cdhit.clstr", clusters, rep_clusters)
print("合并cdhit1结果完成！")
with open("/home/yaoshuai/tools/merge/mergeGenomeCluster.pkl", 'w') as fp:
    pickle.dump(clusters, fp)
print("写入clusters结果成功！")
with open("/home/yaoshuai/tools/merge/mergeGenomeClusterRep.pkl", 'w') as fp:
    pickle.dump(rep_clusters, fp)
print("写入clusters结果成功！")
