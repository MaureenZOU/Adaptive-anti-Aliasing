import torch
import numpy as np

train_dataset = torch.load('/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/train.da')

subset = {}
cnt = 0
for key in train_dataset.keys():
    length = len(train_dataset[key]['image'])
    if length <= 50:
        continue    
    subset[cnt] = train_dataset[key]
    cnt += 1

var_num = [10,20,30,40,50]
out_subset = [{} for i in range(0, len(var_num))]

for i in range(0, len(var_num)):
    for key in subset.keys():
        out_subset[i][key] = {}
        out_subset[i][key]['seq_name'] = subset[key]['seq_name']
        seq_len = len(subset[key]['image'])
        random_id = np.random.randint(0, seq_len, size=var_num[i])
        out_subset[i][key]['image'] = [subset[key]['image'][id] for id in random_id]
        out_subset[i][key]['class'] = [subset[key]['class'][id] for id in random_id]
    torch.save(out_subset[i], '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/train_var' + str(var_num[i]) + '.da')