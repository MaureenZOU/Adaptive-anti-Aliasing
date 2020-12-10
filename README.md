# Delving-Deeper-Into-Anti-Aliasing-in-ConvNets

This work is accepted in **BMVC2020** as ***Best Paper Award***. It introduces a plugin module in neural network to improve both model accuracy and consistency.

\[[Project page](https://maureenzou.github.io/ddac/)\] | \[[arXiv](https://maureenzou.github.io/ddac/)\] | \[[Slide](https://drive.google.com/file/d/1rX_LRfLCwr3nbX3jmpdKlz9L2S8GrrHS/view?usp=sharing)\] | \[[Video](https://www.youtube.com/watch?v=R8eSs6Cljvc)\] | \[[视频](https://www.bilibili.com/video/BV1aD4y127MF/)\]

![alt text](images/tittle.gif)

## Model Zoo

| Model Name                       |            | mIOU | Consistency |
|----------------------------------|------------|-----------|-------------|
| resnet101 | [weight](https://drive.google.com/file/d/1Ls7_u9WStbYcTToI6fscJdk8Tr1kupEn/view?usp=sharing) | 78.5      | 95.5        |
| resnet101 lpf | [weight](https://drive.google.com/file/d/1QMf38efAS8Ddiz-WL-y6ltlPwIuTXoaA/view?usp=sharing) | 78.9      | 95.9        |
| resnet101 gpasa | [weight](https://drive.google.com/file/d/1zSKZMhLJKCQRjyFZTXsMxPF47RJoRhYo/view?usp=sharing) | 79.5      | 96.0        |


## Get started

### Prepare code
Here is the bash script where I previously try to config environment
```
mkdir deeplab-v3plus
cd deeplab-v3plus
mkdir data
cd data
mkdir cityscape
wget xueyan@XXXXXXX:1234/leftImg8bit_trainvaltest.zip
wget xueyan@XXXXXXX:1234/gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
rm -rf license.txt README
unzip gtFine_trainvaltest.zip
rm -rf leftImg8bit_trainvaltest.zip
rm -rf gtFine_trainvaltest.zip
cd ..
mkdir output
mkdir lpf_weights
cd lpf_weights
wget xueyan@XXXXXXX:1124/data/lpf_weights/resnet101_lpf3.pth.tar
cd ..
mkdir checkpoints
cd checkpoints
mkdir resnet101_pasa_group8_softmax_warmup5_old
cd resnet101_pasa_group8_softmax_warmup5_old
wget xueyan@XXXXXXX:1124/data/checkpoints/resnet101_pasa_group8_softmax_warmup5_old/model_best.pth.tar
cd ..
cd ..
cd ..
wget xueyan@XXXXXXX:1124/xueyan-dev.tar.gz
tar -xvzf xueyan-dev.tar.gz
rm -rf xueyan-dev.tar.gz
cd xueyan-dev
```

### link dataset

```bash
cd /PTH/TO/pasa/deeplab-v3plus/zhiding-dev/Deeplab-v3plus/datasets/data/
ln /PTH/TO/cityscapes cityscapes
```

### environment
```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
pip install sklearn visdom
```

### run script

```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
cd deeplab-v3plus/zhiding-dev/Deeplab-v3plus/
python main.py --model deeplabv3plus_lpf_resnet101 --gpu_id 0,1 --dataset cityscapes --year 2012_aug --crop_val --lr 0.00875 --crop_size 768 --batch_size 14 --warmup_iter 8000 --total_itrs 80000 --output_stride 16 -f 3 --out-dir ../../data/output/deeplabv3plus_lpf_resnet101_warmup_8000_80000_cityscape
```
