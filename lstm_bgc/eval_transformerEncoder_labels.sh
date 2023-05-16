#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pt
source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
nohup python -u ./transformer_BGC_labels/eval.py \
    --modelPath ./data/corpus_word2vec.sav \
    --datasetPath ./data/BGC_TD_dataset.csv --seed 42\
    --max_len 128 --nhead 4 --num_encoder_layers 6 --dropout 0.5 \
    --batch_size 32 --name transformerEncoder_labels_test\
    --transformerEncoderPath ./modelSave/transformerEncoder_labels/bS_32_lE_2000_lR_0.0001_mL_128_d_200_nEL_6_dP_0.1/transformerEncoder_Model_labels_2977.pt \
    >nohup_eval_transformerEncoder_labels_labels_dataset.out &