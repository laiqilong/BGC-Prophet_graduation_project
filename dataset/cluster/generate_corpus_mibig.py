import os
import pickle
import multiprocessing
from functools import partial
with open("/mnt/hdd0/qllai/cluster_genome/pro2rep_id.pkl", 'rb') as fp:
    pro2rep_id = pickle.load(fp)

mibig_path = "/mnt/hdd0/public/fasta/"
mibig_ec_path = "/mnt/hdd0/qllai/deepec/output/"
mibig_bgcs = os.listdir(mibig_path)

# Unknown = []
# All_pro2ec = {}
def generate_corpus(genome_faa, Unknown, All_pro2ec):
    # global Unknown
    # global All_pro2ec
    if genome_faa == 'GCF_009914755.1_T2T-CHM13v2.0_protein.faa':
        return []
    with open(mibig_path+genome_faa, 'r') as f:
        proteins = []
        text = f.readlines()
        for line in text:
            if '>' in line:
                protein = line.split()[0][1:]
                proteins.append(protein)
    ecResult = mibig_ec_path+"output"+genome_faa+'/DeepEC_Result.txt'
    pro2ec = {}
    if os.path.exists(ecResult):        
        with open(ecResult, 'r') as f:
            text = f.readlines()
            for line in text:
                if "EC:" in line:
                    kv = line.split()
                    pro2ec[kv[0]] = kv[1]
    ids = []
    for protein in proteins:
        if protein in pro2ec:
            ids.append(pro2ec[protein])
        elif protein in pro2rep_id:
            ids.append(pro2rep_id[protein])
        else:
            ids.append('Unknown')
            Unknown.append(protein)
            print(f'{genome_faa}里的{protein}既没有cluster_id也没有ec号！')
    All_pro2ec.update(pro2ec)
    return ids

def write_corpus(output,lock):
    line = ' '.join(output)
    lock.acquire()
    with open("/mnt/hdd0/qllai/cluster/corpus_br.txt", 'a+') as fp:
        fp.write(line+'\n')
    lock.release()
    lock.acquire()
    with open("/mnt/hdd0/qllai/cluster/corpus.txt", "a+") as fp:
        fp.write(line+' ')
    lock.release()

# def init(l):
#     global lock
#     lock = l
manager1 = multiprocessing.Manager()
Unknown = manager1.list()
All_pro2ec = manager1.dict()
pool = multiprocessing.Pool(processes=28)
partial_generate_corpus = partial(generate_corpus, Unknown=Unknown, All_pro2ec=All_pro2ec)
pool_outputs = pool.map(partial_generate_corpus, mibig_bgcs)
print("任务多发完成！")
pool.close()
print("pool.close()")
pool.join()
print("多线程执行完毕！")

with open('./All_pro2ec_mibig.pkl', 'wb') as fp:
    pickle.dump(dict(All_pro2ec), fp)
print("All_pro2ec_mibig.pkl保存成功！")

with open("./Unknown_mibig.pkl", 'wb') as fp:
    pickle.dump(list(Unknown), fp)
print("Unknown_mibig.pkl写入成功！")

with open('./pool_outputs_mibig.pkl', 'wb') as fp:
    pickle.dump(pool_outputs, fp)
print("pool_outputs_mibig.pkl保存成功！")

# with open("/mnt/hdd0/qllai/cluster/corpus_br.txt", 'w') as fp:
#     fp.write('')
# with open("/mnt/hdd0/qllai/cluster/corpus.txt", "w") as fp:
#     fp.write('')


manager2 = multiprocessing.Manager()
lock = manager2.Lock()
partial_write_corpus = partial(write_corpus, lock = lock)
pool = multiprocessing.Pool(processes=28)
pool.map(partial_write_corpus, pool_outputs)
print("语料库写入完成！")
pool.close()
pool.join()
print("Done!")


