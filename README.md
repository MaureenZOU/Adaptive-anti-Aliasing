# Delving-Deeper-Into-Anti-Aliasing-in-ConvNets

This work is accepted in **BMVC2020** as ***Oral presentation***. It introduces a plugin module in neural network to improve both model accuracy and consistency.

\[[Project page](https://maureenzou.github.io/ddac/)\] | \[[arXiv](https://maureenzou.github.io/ddac/)\] | \[[Slide](https://drive.google.com/file/d/1rX_LRfLCwr3nbX3jmpdKlz9L2S8GrrHS/view?usp=sharing)\] | \[[Video](https://www.youtube.com/watch?v=R8eSs6Cljvc)\] | \[[视频](https://www.bilibili.com/video/BV1aD4y127MF/)\]

![alt text](images/tittle.gif)

## Progress
- [x] Image Classification
- [ ] Instance Segmentation
- [ ] Semantic Segmentation

## Installation
```
torch==1.1.0
torchvision==0.2.0
```

## Dataset
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## File Structure
```
anti-aliasing
└── data
    ├── output
    ├── ILSVRC2012
└── master
    └── Adaptive-anti-Aliasing
        └── ...
```

## Model Zoo

| Model Name                       |            | Top-1 Acc | Consistency |
|----------------------------------|------------|-----------|-------------|
| resnet101_k3_pasa_group8_softmax | [weight]() | 79.0      | 91.8        |
| resnet101_k5_pasa_group8_softmax | [weight]() | 78.6      | 92.2        |

## Testing

```
python main.py --data ../../data/ILSVRC2012 -f 5 -e -b 32 -a resnet101_pasa_group_softmax --group 8 --weights /pth/to/model
```

## Training
```
python main.py --data ../../data/ILSVRC2012 -f 3 -b 128 -ba 2 -a resnet101_pasa_group_softmax --group 8 --out-dir /pth/to/output/dir
```

## Acknowledgement
We borrow most of the code from Richard Zhang's Repo (https://github.com/adobe/antialiased-cnns) Thank you : )
