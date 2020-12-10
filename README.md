

Pretrained weight
```
ResNet 101: https://drive.google.com/file/d/1Ls7_u9WStbYcTToI6fscJdk8Tr1kupEn/view?usp=sharing
ResNet 101, lpf: https://drive.google.com/file/d/1QMf38efAS8Ddiz-WL-y6ltlPwIuTXoaA/view?usp=sharing
ResNet 101, gpasa: https://drive.google.com/file/d/1zSKZMhLJKCQRjyFZTXsMxPF47RJoRhYo/view?usp=sharing
```

Prepare code

```bash
cd /PTH/TO/pasa
wget xueyan@169.237.118.52:1124/deeplab.sh
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

```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
cd deeplab-v3plus/zhiding-dev/Deeplab-v3plus/
python main.py --model deeplabv3plus_gpasa_resnet101 --gpu_id 2,3 --dataset cityscapes --year 2012_aug --crop_val --lr 0.00875 --crop_size 768 --batch_size 14 --warmup_iter 4000 --total_itrs 80000 --output_stride 16 --group 8 -f 3 --out-dir ../../data/output/deeplabv3plus_gpasa_resnet101_warmup_schdule3_4000_80000_cityscape
```

```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
cd deeplab-v3plus/zhiding-dev/Deeplab-v3plus/
python main.py --model deeplabv3plus_lpf_resnet101 --gpu_id 4,5 --dataset cityscapes --year 2012_aug --crop_val --lr 0.00875 --crop_size 768 --batch_size 14 --warmup_iter 4000 --total_itrs 80000 --output_stride 16 -f 3 --out-dir ../../data/output/deeplabv3plus_lpf_resnet101_warmup_schdule3_4000_80000_cityscape
```

```bash
cd /PTH/TO/pasa/
source ./env/pasa/bin/activate
cd deeplab-v3plus/zhiding-dev/Deeplab-v3plus/
python main.py --model deeplabv3plus_resnet101 --gpu_id 6,7 --dataset cityscapes --year 2012_aug --crop_val --lr 0.00875 --crop_size 768 --batch_size 14 --warmup_iter 4000 --total_itrs 80000 --output_stride 16 -f 3 --out-dir ../../data/output/deeplabv3plus_resnet101_warmup_schdule3_4000_80000_cityscape
```
