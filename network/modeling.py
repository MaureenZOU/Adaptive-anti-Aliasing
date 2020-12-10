from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet, resnet_gpasa, resnet_lpf, resnet_pasa, resnet_pasa_debug, resnet_lpf_debug
from .backbone import mobilenetv2
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': None,
}

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_gpasa_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True, **kwargs):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = resnet_gpasa.__dict__[backbone_name[6:]](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_pasa_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True, **kwargs):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = resnet_pasa.__dict__[backbone_name[5:]](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_pasadebug_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True, **kwargs):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = resnet_pasa_debug.__dict__[backbone_name[10:]](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_lpf_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True, **kwargs):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = resnet_lpf.__dict__[backbone_name[4:]](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_lpfdebug_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True, **kwargs):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = resnet_lpf_debug.__dict__[backbone_name[9:]](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone=True):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, pretrained, progress, num_classes, output_stride=8, **kwargs):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    elif backbone.startswith('gpasa_resnet'):
        model = _segm_gpasa_resnet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    elif backbone.startswith('pasa_resnet'):
        model = _segm_pasa_resnet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    elif backbone.startswith('pasadebug_resnet'):
        model = _segm_pasadebug_resnet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    elif backbone.startswith('lpf_resnet'):
        model = _segm_lpf_resnet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    elif backbone.startswith('lpfdebug_resnet'):
        model = _segm_lpfdebug_resnet(arch_type, backbone, num_classes, output_stride=output_stride, **kwargs)
    else:
        raise NotImplementedError

    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model

def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, **kwargs)

def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, **kwargs)

def deeplabv3plus_resnet50(pretrained=False, progress=True,
                       num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'resnet50', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_gpasa_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'gpasa_resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_pasa_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'pasa_resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_pasadebug_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'pasadebug_resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_lpf_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'lpf_resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_lpfdebug_resnet101(pretrained=False, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'lpfdebug_resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3_mobilenet(pretrained=False, progress=True,
                       num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'mobilenetv2', pretrained, progress, num_classes, **kwargs)


def deeplabv3plus_mobilenet(pretrained=False, progress=True,
                       num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', pretrained, progress, num_classes, **kwargs)
