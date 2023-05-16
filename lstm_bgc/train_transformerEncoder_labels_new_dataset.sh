#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
# source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch
nohup python -u ./transformer_BGC_labels/trainer.py \
    --modelPath ./data/corpus_word2vec.sav \
    --datasetPath ./data/BGC_TD_dataset_10.csv --seed 42\
    --max_len 128 --nhead 4 --num_encoder_layers 6 --dropout 0.1 \
    --batch_size 32 --learning_rate 0.01 --label_epochs 2000 \
    >nohup_transformerEncoder_labels_new_dataset.out &