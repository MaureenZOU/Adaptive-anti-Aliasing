## Get started


## Model Zoo

| Model Name                       |            | mIOU | Consistency |
|----------------------------------|------------|-----------|-------------|
| resnet101 | [weight](https://drive.google.com/file/d/1Ls7_u9WStbYcTToI6fscJdk8Tr1kupEn/view?usp=sharing) | 78.5      | 95.5        |
| resnet101 lpf | [weight](https://drive.google.com/file/d/1QMf38efAS8Ddiz-WL-y6ltlPwIuTXoaA/view?usp=sharing) | 78.9      | 95.9        |
| resnet101 gpasa | [weight](https://drive.google.com/file/d/1zSKZMhLJKCQRjyFZTXsMxPF47RJoRhYo/view?usp=sharing) | 79.5      | 96.0        |


Prepare code

```bash
cd /PTH/TO/pasa

sh deeplab.sh
```

link dataset

```bash
cd /PTH/TO/pasa/deeplab-v3plus/zhiding-dev/Deeplab-v3plus/datasets/data/
ln /PTH/TO/cityscapes cityscapes
```

environment
```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
pip install sklearn visdom
```

run script

```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
cd deeplab-v3plus/zhiding-dev/Deeplab-v3plus/
python main.py --model deeplabv3plus_lpf_resnet101 --gpu_id 0,1 --dataset cityscapes --year 2012_aug --crop_val --lr 0.00875 --crop_size 768 --batch_size 14 --warmup_iter 8000 --total_itrs 80000 --output_stride 16 -f 3 --out-dir ../../data/output/deeplabv3plus_lpf_resnet101_warmup_8000_80000_cityscape
```
