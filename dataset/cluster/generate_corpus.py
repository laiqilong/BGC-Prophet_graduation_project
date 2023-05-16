import os
import pickle
import multiprocessing

with open("/mnt/hdd0/qllai/cluster_genome/pro2rep_id.pkl", 'rb') as fp:
    pro2rep_id = pickle.load(fp)

genome_path = "/mnt/hdd0/public/Bac_Refseq_20230225/protein_faa_decompress/"
ecpred_path = '/mnt/hdd0/public/deepecpred/'
genomes = os.listdir(genome_path)

Unknown = []
All_pro2ec = {}
def generate_corpus(genome_faa):
    with open(genome_path+genome_faa, 'r') as f:
        proteins = []
        text = f.readlines()
        for line in text:
            if '>' in line:
                protein = line.split()[0][1:]
                proteins.append(protein)
    ecResult = ecpred_path+genome_faa.replace('.faa', '')+'/DeepEC_Result.txt'
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
            print('f{genome_faa}里的{protein}既没有cluster_id也没有ec号！')
    All_pro2ec.update(pro2ec)
    return ids
            
pool = multiprocessing.Pool(processes=28)
pool_outputs = pool.map(generate_corpus, genomes)
pool.close()
pool.join()
corpus_br = ''
corpus = ''
with open('./pool_outputs.pkl', 'wb') as fp:
    pickle.dump(pool_outputs, fp)
print("pool_outputs.pkl保存成功！")

for output in pool_outputs:
    line = ' '.join(output)
    corpus_br = corpus_br + line +'\n'
    corpus = corpus + line + ' '

with open('./All_pro2ec.pkl', 'wb') as fp:
    pickle.dump(All_pro2ec, fp)
print("All_pro2ec.pkl保存成功！")

with open("./corpus_br.txt", 'w') as fp:
    fp.write(corpus_br)
print("corpus_br.txt写入成功！")

with open("./corpus.txt", 'w') as fp:
    fp.write(corpus)
print("corpus.txt写入成功！")

with open("./Unknown.pkl", 'w') as fp:
    fp.write(Unknown)
print("Unknown.pkl写入成功！")