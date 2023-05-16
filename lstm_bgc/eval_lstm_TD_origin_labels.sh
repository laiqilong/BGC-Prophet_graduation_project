#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
nohup python -u ./lstm_bgc_everygene/eval.py \
    --modelPath ./data/corpus_word2vec.sav \
    --datasetPath ./data/NineGenomes_128.csv \
    --max_len 128 --hidden_dim 256 --num_layers 2 --dropout 0.5 \
    --batch_size 64 --name NineGm_TD_origin_128_dataset\
    --lstmPath ./modelSave/multiheads_attention_bS_64_lE_5_lR_0.001_mL_128_hD_256_nL_2_dP_0.5/LSTM_Model_TD_37.pt \
    >nohup_eval_lstm_NineGm_TD_origin_128_dataset.out &
