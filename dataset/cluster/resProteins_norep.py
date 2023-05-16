import sys
import re
path = sys.argv[1]
path_norep = sys.argv[2]
with open(path, 'r') as f:
    proteins = {}
    text = f.read()
    seqPattern = re.compile('>')
    seq_start = [substr.start() for substr in re.finditer(seqPattern, text)] + [len(text)]
    for i in range(len(seq_start)-1):
        seq_block = text[seq_start[i]:seq_start[i+1]]
        seq_lines = seq_block.split('\n')
        protein_name = seq_lines[0][1:] # 只保留蛋白的AC号
        # 去除>
        protein_seq = seq_lines[1:]
        protein_seq = ''.join(protein_seq).replace('\n', '')
        proteins[protein_name] = protein_seq

with open(path_norep, 'w') as f:
    for pro, seq in proteins():
        proBlock = '>' + pro + '\n' + seq + '\n'
        f.write(proBlock)