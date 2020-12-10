## Model Zoo

| Model Name                       |            | mIOU | Consistency |
|----------------------------------|------------|-----------|-------------|
| resnet101 | [weight](https://drive.google.com/file/d/1oky8pbqHiINit9-0Ybu-JZQdZkEIUxry/view?usp=sharing) | 79.0      | 91.8        |
| resnet101 lpf | [weight](https://drive.google.com/file/d/1rfZ2-W7NM0CfmxkMIxrMAhIgWGBNDMwI/view?usp=sharing) | 78.6      | 92.2        |
| resnet101 ours | [weight](https://drive.google.com/file/d/1rfZ2-W7NM0CfmxkMIxrMAhIgWGBNDMwI/view?usp=sharing) | 78.6      | 92.2        |

Pretrained weight
```
ResNet 101: https://drive.google.com/file/d/1Ls7_u9WStbYcTToI6fscJdk8Tr1kupEn/view?usp=sharing
ResNet 101, lpf: https://drive.google.com/file/d/1QMf38efAS8Ddiz-WL-y6ltlPwIuTXoaA/view?usp=sharing
ResNet 101, gpasa: https://drive.google.com/file/d/1zSKZMhLJKCQRjyFZTXsMxPF47RJoRhYo/view?usp=sharing
```

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
