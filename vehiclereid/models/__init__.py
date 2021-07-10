from __future__ import absolute_import

from .resnet import *
from .glamor import *
from .resnet_cbam import *
from .glamor_v1 import *
from .glamor_nv1 import *
from .glamor_v2 import *
from .glamor_v3 import *
from .resnext import *
from .residual_attention_network import *


__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
    'glamor50': glamor50,
    'glamor18': glamor18,
    'resnet50_cbam': resnet50_cbam,
    'glamor50_v1': glamor50_v1,
    'glamor50_nv1': glamor50_nv1,
    'glamor50_v2': glamor50_v2,
    'glamor50_v3': glamor50_v3,
    'resnext101': resnext101_64x4d,
    'resnet56_attention': resnet56_attention,
    'resnet92_attention': resnet92_attention
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
