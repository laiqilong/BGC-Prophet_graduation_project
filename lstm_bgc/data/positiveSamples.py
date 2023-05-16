import pickle

with open('/home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/BGC2line.pkl', 'rb') as fp:
    BGC2line = pickle.load(fp)

with open('/home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/mibig_labels.pkl', 'rb') as fp:
    mibig_labels = pickle.load(fp)

print(BGC2line)
BGC_labels_dataset = {}
for k, v in BGC2line.items():
    BGC_labels_dataset[k] = (v, mibig_labels[k])

with open('/home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/BGC_labels_dataset.pkl', 'wb') as fp:
    pickle.dump(BGC_labels_dataset, fp)