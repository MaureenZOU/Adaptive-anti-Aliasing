import glob
import xml.etree.ElementTree as ET
from shutil import copyfile
import os
import torch

# annot_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015/Annotations/VID/train/'
# out_dir = '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet'

# img_seq_dict = {}
# cnt = 0
# files = sorted(glob.glob(annot_root + 'ILSVRC2015*/*/*.xml'))
# # files = sorted(glob.glob(annot_root + 'ILSVRC201*/*.xml'))

# for file in files:
#     seq_name = file.split('/')[10]
#     image_pth = '/'.join(file.split('/')[6:]).replace('Annotations', 'Data').replace('.xml', '.JPEG')

#     class_ = []
#     tree = ET.parse(file)
#     root = tree.getroot()
#     for obj in root.iter('name'):
#         class_.append(obj.text)
    
#     if len(class_) == 0:
#         continue

#     if seq_name not in img_seq_dict.keys():
#         img_seq_dict[seq_name] = {'image':[], 'class':[]}
    
#     img_seq_dict[seq_name]['image'].append(image_pth)
#     img_seq_dict[seq_name]['class'].append(list(set(class_)))
#     cnt += 1

#     if cnt % 1000 == 0:
#         print(cnt, len(files))

# img_seqid_dict = {}
# seq_id = 0
# for key in img_seq_dict.keys():
#     img_seqid_dict[seq_id] = img_seq_dict[key]
#     img_seqid_dict[seq_id]['seq_name'] = key
#     seq_id += 1

# torch.save(img_seqid_dict, os.path.join(out_dir, 'train.da'))

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

# annot_pth = '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/ILSVRC2015-imagenet/cls_to_id.da'
# cls_to_id = {}
# for i in range(0, len(class_lst)):
#     cls_to_id[class_lst[i]] = i

# torch.save(cls_to_id, annot_pth)

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


# annot_root = '/home/xueyan/antialias-cnn/data/ILSVRC2015/Annotations/VID/val/'
# out_dir = '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet'

# glob_lst = glob.glob(annot_root + 'ILSVRC201*')
# img_seq_dict = {}
# cnt = 0
# for folder in glob_lst:
#     file_names = sorted(glob.glob(folder + '/*.xml'))

#     # for file in [file_names[(len(file_names)*2)//5], file_names[(len(file_names)*1)//5], file_names[(len(file_names)*3)//5]]:
#     for index in [(len(file_names)*2)//5, (len(file_names)*1)//5, (len(file_names)*3)//5]:
#         while index < len(file_names):
#             file = file_names[index]
#             file_name = '/'.join(file.split('/')[6:]).replace('Annotations', 'Data').replace('.xml', '.JPEG')
#             seq_name = file.split('/')[9]
#             tree = ET.parse(file)
#             root = tree.getroot()    
#             class_ = []
#             for obj in root.iter('name'):
#                 class_name = obj.text
#                 class_.append(class_name)

#             if len(class_) == 0:
#                 index += 1
#                 continue

#             img_seq_dict[cnt] = {}
#             img_seq_dict[cnt]['image'] = []
#             img_seq_dict[cnt]['class'] = []

#             img_seq_dict[cnt]['seq_name'] = seq_name
#             img_seq_dict[cnt]['image'].append(file_name)
#             img_seq_dict[cnt]['class'].append(list(set(class_)))

#             cnt += 1
#             break

#         if cnt % 1000 == 0:
#             print(cnt*1.0 / (len(glob_lst)*3))

# print(img_seq_dict)
# torch.save(img_seq_dict, os.path.join(out_dir, 'val.da'))
# print('done')

