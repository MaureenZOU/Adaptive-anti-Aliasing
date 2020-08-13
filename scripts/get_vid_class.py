import glob
import xml.etree.ElementTree as ET
from shutil import copyfile
import os

# annot_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015/Annotations/VID/train/'
# class_ = set()

# print(annot_root + 'ILSVRC2015*/*/*.xml')
# for file in glob.glob(annot_root + 'ILSVRC2015*/*/*.xml'):
#     tree = ET.parse(file)
#     root = tree.getroot()
#     for obj in root.iter('name'):
#         class_.add(obj.text)

# print(class_)
# print(len(class_))

# '''
# 'n02402425', 'n02062744', 'n01674464', 'n02958343', 'n02419796', 'n02118333', 'n04530566', 'n02924116', 'n02129165', 'n01503061', 
# 'n02131653', 'n02503517', 'n02834778', 'n02391049', 'n02121808', 'n04468005', 'n02484322', 'n03790512', 'n01726692', 'n01662784', 
# 'n02374451', 'n02355227', 'n02411705', 'n02691156', 'n02342885', 'n02129604', 'n02510455', 'n02509815', 'n02084071', 'n02324045'
# '''

# import os
# class_lst = ['n02402425', 'n02062744', 'n01674464', 'n02958343', 'n02419796', 'n02118333', 'n04530566', 'n02924116', 'n02129165', 'n01503061', 
#              'n02131653', 'n02503517', 'n02834778', 'n02391049', 'n02121808', 'n04468005', 'n02484322', 'n03790512', 'n01726692', 'n01662784',
#              'n02374451', 'n02355227', 'n02411705', 'n02691156', 'n02342885', 'n02129604', 'n02510455', 'n02509815', 'n02084071', 'n02324045']

# root = '/home/xueyan/antialias-cnn/data/ILSVRC2015-Imagenet/train_mini'
# for name in class_lst:
#     os.makedirs(os.path.join(root, name), exist_ok=True)

# annot_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015/Annotations/VID/val/'
# tar_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015-Imagenet/val/'

# glob_lst = glob.glob(annot_root + 'ILSVRC201*/*.xml')
# cnt = 0 
# for file in glob_lst:
#     file_name = file.replace('Annotations', 'Data').replace('xml', 'JPEG')
#     tree = ET.parse(file)
#     root = tree.getroot()    
#     for obj in root.iter('name'):
#         class_name = obj.text
#         tar_file_name = file_name.split('/')[9] + '-' + file_name.split('/')[10]
#         copyfile(file_name, os.path.join(tar_root, class_name, tar_file_name))
#     if cnt % 1000 == 0:
#         print(cnt*1.0 / len(glob_lst))
#     cnt += 1
    
# print('done')

# annot_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015/Annotations/VID/val/'
# tar_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015-Imagenet/val_mini/'

# glob_lst = glob.glob(annot_root + 'ILSVRC201*')
# cnt = 0
# for folder in glob_lst:
#     file_names = sorted(glob.glob(folder + '/*.xml'))
#     for file in [file_names[(len(file_names)*2)//3], file_names[(len(file_names)*1)//3]]:
#         file_name = file.replace('Annotations', 'Data').replace('xml', 'JPEG')
#         tree = ET.parse(file)
#         root = tree.getroot()    
#         for obj in root.iter('name'):
#             class_name = obj.text
#             tar_file_name = file_name.split('/')[9] + '-' + file_name.split('/')[10]
#             copyfile(file_name, os.path.join(tar_root, class_name, tar_file_name))
#         if cnt % 1000 == 0:
#             print(cnt*1.0 / len(glob_lst))
#         cnt += 1    
# print('done')

annot_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015/Annotations/VID/train/'
tar_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015-Imagenet/train_mini/'

glob_lst = glob.glob(annot_root + '*/ILSVRC201*')
cnt = 0
for folder in glob_lst:
    file_names = sorted(glob.glob(folder + '/*.xml'))
    for file in [file_names[(len(file_names)*2)//5], file_names[(len(file_names)*1)//5], file_names[(len(file_names)*3)//5]]:
        file_name = file.replace('Annotations', 'Data').replace('xml', 'JPEG')
        tree = ET.parse(file)
        root = tree.getroot()    
        for obj in root.iter('name'):
            class_name = obj.text
            tar_file_name = file_name.split('/')[9] + '-' + file_name.split('/')[10] + '-' + file_name.split('/')[11]
            copyfile(file_name, os.path.join(tar_root, class_name, tar_file_name))
        if cnt % 1000 == 0:
            print(cnt*1.0 / (len(glob_lst)*3))
        cnt += 1    
print('done')
