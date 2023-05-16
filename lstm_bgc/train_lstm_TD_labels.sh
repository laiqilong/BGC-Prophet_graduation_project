#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
nohup python -u ./lstm_bgc_everygene/trainer.py \
    --modelPath ./data/corpus_word2vec.sav \
    --num_heads 4 --depth 2\
    --datasetPath ./data/BGC_positive_dataset_no_insert_10.csv --seed 42\
    --max_len 128 --hidden_dim 256 --num_layers 2 --dropout 0.5 \
    --batch_size 64 --learning_rate 0.001 --label_epochs 5 --distribute_epochs 1000\
    >nohup_lstm_TD_labels_new_dataset.out &