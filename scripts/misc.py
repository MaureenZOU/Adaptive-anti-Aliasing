import torch
import json

# with open('/home/xueyan/robust-inst-seg/data/imagenet-vid-robust/misc/imagenet_vid_class_index.json') as json_file:
#     imagenet_vid_class_index = json.load(json_file)

# with open('/home/xueyan/robust-inst-seg/data/imagenet-vid-robust/misc/vid_class_name_to_imagenet_idx.json') as json_file:
#     vid_class_name_to_imagenet_idx = json.load(json_file)

# vid_cls_to_name = {}
# for key in imagenet_vid_class_index:
#     vid_cls_to_name[imagenet_vid_class_index[key][1]] = imagenet_vid_class_index[key][0]

# vid_cls_to_imagenet_id = {}
# for key in vid_class_name_to_imagenet_idx:
#     vid_cls_to_imagenet_id[vid_cls_to_name[key]] = [int(x) for x in vid_class_name_to_imagenet_idx[key]]

# torch.save(vid_cls_to_imagenet_id, '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/vid_cls_to_imagenet_id.da')


# with open('/home/xueyan/robust-inst-seg/data/imagenet-vid-robust/misc/rev_wnid_map.json') as json_file:
#     wnid_map = json.load(json_file)

# with open('/home/xueyan/robust-inst-seg/data/imagenet-vid-robust/misc/imagenet_class_index.json') as json_file:
#     imagenet_class_index = json.load(json_file)

# imagenet_cls_to_idx = {}

# for key in imagenet_class_index:
#     imagenet_cls_to_idx[imagenet_class_index[key][0]] = int(key)

# out = {}
# for key in wnid_map:
#     out[key] = [imagenet_cls_to_idx[x] for x in wnid_map[key]]

# torch.save(out, '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/vid_cls_to_imagenet_id.da')
# print(out)


# with open('/home/xueyan/robust-inst-seg/data/imagenet-vid-robust/misc/rev_wnid_map.json') as json_file:
#     wnid_map = json.load(json_file)

# with open('/home/xueyan/robust-inst-seg/data/imagenet-vid-robust/misc/imagenet_class_index.json') as json_file:
#     imagenet_class_index = json.load(json_file)

# imagenet_cls_to_idx = {}
# for key in imagenet_class_index:
#     imagenet_cls_to_idx[imagenet_class_index[key][0]] = int(key)

# imagenet_classes = []
# for key in wnid_map:
#     for cls_ in wnid_map[key]:
#         imagenet_classes.append(imagenet_cls_to_idx[cls_])

# x = torch.zeros((1000))
# x[imagenet_classes] = 1

# torch.save(x, '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/valid_class.da')

# out = {}
# for key in wnid_map:
#     out[key] = [imagenet_cls_to_idx[x] for x in wnid_map[key]]

# torch.save(out, '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/vid_cls_to_imagenet_id.da')
# print(out)

# with open('/home/xueyan/antialias-cnn/data/imagenet-vid-robust/metadata/pmsets.json') as json_file:
#     imagenet_class_index = json.load(json_file)

with open('/home/xueyan/antialias-cnn/data/imagenet-vid-robust/metadata/labels.json') as json_file:
    labels = json.load(json_file)

with open('/home/xueyan/antialias-cnn/data/imagenet-vid-robust/misc/imagenet_vid_class_index.json') as json_file:
    imagenet_vid_class_index = json.load(json_file)

# # print(imagenet_class_index)
# # print(labels)
# # print(torch.load('/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/val.da'))

imagenet_vid_idx_to_cls = {}
for key in imagenet_vid_class_index:
    imagenet_vid_idx_to_cls[int(key)] = imagenet_vid_class_index[key][0]

# cnt = 0
# out = {}
# for key in imagenet_class_index:
#     out[cnt] = {}
#     cls_ = [[imagenet_vid_idx_to_cls[x] for x in labels[key]]]
#     img_name = 'Data/VID/' + key
#     seq_name = key.split('/')[1]
#     out[cnt]['image'] = [img_name]
#     out[cnt]['seq_name'] = seq_name
#     out[cnt]['class'] = cls_
#     cnt += 1

# torch.save(out, '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/val_anchor.da')
# print(out)

with open('/home/xueyan/antialias-cnn/data/imagenet-vid-robust/metadata/pmsets.json') as json_file:
    pmsets = json.load(json_file)

out = {}
cnt = 0
for key in pmsets.keys():
    img_names = []
    labels_ = []
    img_names.append('Data/VID/' + key)
    seq_name = key.split('/')[1]
    cls_ = [imagenet_vid_idx_to_cls[int(x)] for x in labels[key]]
    labels_.append(cls_)

    if len(pmsets[key]) < 20:
        continue

    out[cnt] = {}
    for vid in pmsets[key]:
        img_names.append('Data/VID/' + vid)
        cls_ = [imagenet_vid_idx_to_cls[int(x)] for x in labels[vid]]
        labels_.append(cls_)

    out[cnt]['image'] = img_names
    out[cnt]['seq_name'] = seq_name
    out[cnt]['class'] = labels_

    cnt += 1

print(len(out.keys()))
torch.save(out, '/home/xueyan/antialias-cnn/data/ILSVRC2015/py_annot/imagenet/val_robust_20.da')

# print(imagenet_vid_class_index)
# for key in imagenet_class_index:
#     # cls_ = 
#     print(labels[key])