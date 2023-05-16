import lmdb
import pickle
from tqdm import tqdm

Negatives = []
# import pickle
for i in range(0, 120):
    with open(f'./Negatives/output_negatives_{i}.pkl','rb')as fp:
        Negatives += pickle.load(fp)
        print("已处理", i)


lmdb_path = './lmdb_negatives'
map_size = 307374182400
db = lmdb.open(lmdb_path, subdir=True, map_size=map_size, readonly=False, meminit=False, map_async=True)
txn = db.begin(write=True)
for i, negative in tqdm(enumerate(Negatives), desc='Negative lmdb', leave=True):
    txn.put(str(i).encode('ascii'), pickle.dumps(negative))
    if i%100000 == 0:
        # print(i, "commit!")
        txn.commit()
        txn = db.begin(write=True)
txn.commit()
print("Done!")