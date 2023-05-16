from functools import partial

class Protein:
    
    def __init__(self, pathOfFaa, pathOfECPredicted):
        self.pathOfFaa = pathOfFaa
        self.pathOfECPredicted = pathOfECPredicted
    
    def __getECinfo(self):
        with open(self.pathOfECPredicted, "r") as f:
            proteins = {}
            for line in f:
                if "EC" in line:
                    protein = line.split()
                    protein_name, protein_EC = protein[0], protein[1]
                    proteins[protein_name] = protein_EC
        self.ECproteins = proteins
    
    def __getSeqInfo(self):
        with open(self.pathOfFaa, "r") as f:
            import re
            text = f.readlines()
            text = ''.join(text)
            proteins = {}
            seqPattern = re.compile(r'>')
            seq_start = [substr.start() for substr in re.finditer(seqPattern, text)] + [len(text)]
            for i in range(len(seq_start)-1):
                seq_block = text[seq_start[i]:seq_start[i+1]]
                seq_lines = seq_block.split('\n')
                protein_name = seq_lines[0][1:].split()[0] # 只保留蛋白的AC号
                # 去除>
                protein_seq = seq_lines[1:]
                protein_seq = ''.join(protein_seq).replace('\n', '')
                proteins[protein_name] = protein_seq
        self.allProteins = proteins
    
    def __getRes(self):
        self.__getECinfo()
        self.__getSeqInfo()
        allProteinsSet = set(self.allProteins.keys()) # 含有基因组中所有蛋白的AC号
        ECProteinsSet = set(self.ECproteins.keys())
        resProteinsSet = allProteinsSet - ECProteinsSet
        self.resProteinsSet = resProteinsSet
        self.resProteins = {}
        for pro in iter(self.resProteinsSet):
            self.resProteins[pro] = self.allProteins[pro]
        
    def getResSeq(self):
        self.__getRes()
        seqs = ''
        for pro, seq in self.resProteins.items():
            proBlock = '>' + pro + '\n' + seq + '\n'
            seqs += proBlock
        return seqs

import multiprocessing

Allseq = ''
def getOneSeq(proName):
    FaaPath = allProteinsSeqPath + proName + '.faa'
    ECPath = predECsPath + proName + '/DeepEC_Result.txt'
    pro = Protein(FaaPath, ECPath)
    seq = ''
    try:
        seq = pro.getResSeq()
        if len(seq)==0:
            print(FaaPath)
            print(ECPath)
    except:
        print(FaaPath)
        print(ECPath)
        print("获取错误！")
    finally:
        lock.acquire()
        with open("/mnt/hdd0/public/resProteins.fasta", "a+") as f:
            f.write(''.join(seq))
        lock.release()

def init(l):
    global lock
    lock = l

import os
predECsPath = "/mnt/hdd0/public/deepecpred/"
allProteinsSeqPath = "/mnt/hdd0/public/Bac_Refseq_20230225/protein_faa_decompress/"
all_protein = os.listdir(predECsPath)[:1000]
lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes = 20, initializer=init, initargs=(lock, ))
pool_outputs = pool.map(getOneSeq, all_protein)
pool.close()
pool.join()
