#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
nohup python -u ./lstm_bgc_labels/eval.py \
    --modelPath /home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav \
    --datasetPath ./data/BGC_TD_dataset.csv \
    --lstmPath /home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/modelSave/32_500_0.001_64_256_1_0.5/LSTM_Model_128.pt \
    --max_len 64 --hidden_dim 256 --num_layers 1 --dropout 0.5 \
    --batch_size 32 \
    >nohup_lstm_labels_test.out &