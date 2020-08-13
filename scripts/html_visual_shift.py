from utils import futils, Visualizer
import glob
import os

visual_dir = '/home/xueyan/antialias-cnn/data/output/resnet101_ori/visual'
files = glob.glob(os.path.join(visual_dir, '*/*.*'))

class_dict = {}
for pth in files:
    class_name = pth.split('/')[8]
    if class_name not in class_dict.keys():
        class_dict[class_name] = []
    class_dict[class_name].append(pth)

for key in class_dict.keys():
    visualizer = Visualizer(os.path.join(visual_dir, key), demo_name='index.html')
    for pth in class_dict[key]:
        visual_pth = '/'.join(pth.split('/')[-1:])
        visual_name = pth.split('/')[-1]
        visualizer.insert(visual_pth, visual_name)
    visualizer.write()