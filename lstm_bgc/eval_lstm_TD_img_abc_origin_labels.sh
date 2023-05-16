#!/bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
nohup python -u ./lstm_bgc_everygene/eval.py \
    --modelPath /home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav \
    --datasetPath /home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/imgABC_origin_data.csv \
    --max_len 128 --hidden_dim 256 --num_layers 2 --dropout 0.5 \
    --batch_size 32 --name IMG_ABC_TD_origin_dataset\
    --lstmPath /home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/modelSave/bS_32_lE_100_lR_0.001_mL_128_hD_256_nL_2_dP_0.5/LSTM_Model_TD_27.pt \
    >nohup_eval_lstm_IMG_ABC_TD_origin_dataset.out &