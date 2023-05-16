#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pt
source /mnt/hdd0/qllai/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
nohup python -u ./transformer_BGC_TD/eval.py \
    --modelPath ./data/corpus_word2vec.sav \
    --datasetPath ./data/NineGm_TD_origin_dataset.csv \
    --max_len 128 --nhead 4 --num_encoder_layers 6 --dropout 0.5 \
    --batch_size 32 --name NineGm_transformerEncoder_TD_origin_dataset\
    --transformerEncoderPath ./modelSave/transformerEncoder_TD/bS_32_dE_1200_lR_0.01_mL_128_d_200_nEL_6_dP_0.2_TD/transformerEncoder_Model_TD_1199.pt \
    >nohup_eval_transformerEncoder_NineGm_TD_origin_dataset.out &