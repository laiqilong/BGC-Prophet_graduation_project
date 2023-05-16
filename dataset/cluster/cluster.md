# Cluster

## cd-hit

```bash
./cd-hit -i mibig.fasta -o ./output/mibig_cdhit -s 0.80 -c 0.80 -d 25 -M 16000 -T 16
```



## mmseqs2

```bash
mmseqs easy-cluster mibig_cdhit.fasta easy_MIBIG_cdhit tmp/ -s 7.5 -c 0.5
```

