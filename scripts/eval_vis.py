import numpy
import re
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

pths = ['/home/xueyan/antialias-cnn/data/checkpoints/resnet18_lpf_full_01/log.txt', '/home/xueyan/antialias-cnn/data/checkpoints/resnet18_lpf_lspa_full_01/log.txt']
label_list = []

axes = plt.gca()
axes.set_ylim([20, 60])

for pth in pths:
    file = open(pth,'r')
    acc = {}
    for line in file.readlines():
        if 'epoch' in line:
            epoch = int(re.search('epoch: (.*), top-1', line).group(1))
            top1 = float(re.search('top-1 acc(.*)top-5 acc', line).group(1).split('(')[1].split(',')[0])
            acc[epoch] = top1
    
    x = []
    y = []
    for key in sorted(acc.keys()):
        x.append(key)
        y.append(100 - acc[key])
    
    plt.plot(x, y)
    label_list.append(pth.split('/')[-2])

plt.legend(label_list, ncol=1, loc='upper right');
plt.savefig('top1-acc.png')