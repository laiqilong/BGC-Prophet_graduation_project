#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
nohup python -u ./lstm_bgc_labels/trainer.py \
    --modelPath /home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav \
    --datasetPath ./data/BGC_labels_dataset.csv \
    --max_len 64 --hidden_dim 256 --num_layers 1 --dropout 0.5 \
    --batch_size 32 --learning_rate 0.001 --epochs 500 \
    >nohup_lstm_labels.out &