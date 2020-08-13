import torchvision.models as models

import models_lpf.resnet
import models_lpf.resnet_pasa_group_softmax
import models_lpf.resnet_ori

def build_model(model_name, args, summarywriter=None):
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))        

        if(args.arch=='resnet18_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnet18(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)
        elif(args.arch=='resnet34_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnet34(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)
        elif(args.arch=='resnet50_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnet50(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)
        elif(args.arch=='resnet101_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnet101(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)
        elif(args.arch=='resnet152_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnet152(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)
        elif(args.arch=='resnext50_32x4d_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnext50_32x4d(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)
        elif(args.arch=='resnext101_32x8d_pasa_group_softmax'):
            model = models_lpf.resnet_pasa_group_softmax.resnext101_32x8d(filter_size=args.filter_size, pasa_group=args.group, num_classes=args.num_classes)


        elif(args.arch=='resnet18_lpf'):
            model = models_lpf.resnet.resnet18(filter_size=args.filter_size, num_classes=args.num_classes)
        elif(args.arch=='resnet34_lpf'):
            model = models_lpf.resnet.resnet34(filter_size=args.filter_size, num_classes=args.num_classes)
        elif(args.arch=='resnet50_lpf'):
            model = models_lpf.resnet.resnet50(filter_size=args.filter_size, num_classes=args.num_classes)
        elif(args.arch=='resnet101_lpf'):
            model = models_lpf.resnet.resnet101(filter_size=args.filter_size, num_classes=args.num_classes)
        elif(args.arch=='resnet152_lpf'):
            model = models_lpf.resnet.resnet152(filter_size=args.filter_size, num_classes=args.num_classes)
        elif(args.arch=='resnext50_32x4d_lpf'):
            model = models_lpf.resnet.resnext50_32x4d(filter_size=args.filter_size, num_classes=args.num_classes)
        elif(args.arch=='resnext101_32x8d_lpf'):
            model = models_lpf.resnet.resnext101_32x8d(filter_size=args.filter_size, num_classes=args.num_classes)


        elif(args.arch=='resnet18_ori'):
            model = models_lpf.resnet_ori.resnet18(num_classes=args.num_classes)
        elif(args.arch=='resnet34_ori'):
            model = models_lpf.resnet_ori.resnet34(num_classes=args.num_classes)
        elif(args.arch=='resnet50_ori'):
            model = models_lpf.resnet_ori.resnet50(num_classes=args.num_classes)
        elif(args.arch=='resnet101_ori'):
            model = models_lpf.resnet_ori.resnet101(num_classes=args.num_classes)
        elif(args.arch=='resnet152_ori'):
            model = models_lpf.resnet_ori.resnet152(num_classes=args.num_classes)
        elif(args.arch=='resnext50_32x4d_ori'):
            model = models_lpf.resnet_ori.resnext50_32x4d(num_classes=args.num_classes)
        elif(args.arch=='resnext101_32x8d_ori'):
            model = models_lpf.resnet_ori.resnext101_32x8d(num_classes=args.num_classes)

    return model