# Delving-Deeper-Into-Anti-Aliasing-in-ConvNets

This work introduce a plugin module in neural network to improve both model accuracy and consistency.

\[[Project page](https://duckduckgo.com)\] | \[[arXiv](https://duckduckgo.com)\] | \[[slide](https://duckduckgo.com)\] | \[[video](https://duckduckgo.com)\]

![alt text](images/tittle.gif)

## Progress
- [x] Image Classification
- [ ] Instance Segmentation
- [ ] Semantic Segmentation

## Installation

## Dataset

## File Structure

## Model Zoo

## Testing

```
python main.py --data ../../data/ILSVRC2012 -f 5 -e -b 32 -a resnet101_pasa_group_softmax --group 8 --weights /pth/to/model
```

## Training
```
python main.py --data ../../data/ILSVRC2012 -f 3 -b 128 -ba 2 -a resnet101_pasa_group_softmax --group 8 --out-dir /pth/to/output/dir
```
