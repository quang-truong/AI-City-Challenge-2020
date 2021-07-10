from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo

__all__ = ['glamor18_v2', 'glamor50_v2']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, planes):
        super(GlobalAttention, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride= 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride= 1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(inplace= True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = self.conv1(x)
        f = self.lrelu(f)
        f = self.conv2(f)
        f = self.sigmoid(f)
        x = f * x
        return x

class LocalAttention(nn.Module):
    def __init__(self, planes):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride= 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride= 1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(inplace= True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class Branch(nn.Module):
    def __init__(self, planes, block, layers, last_stride):
        super(Branch, self).__init__()
        self.inplanes = 256
        self.remain_bottlenecks = self._remain_bottlenecks(block, 64, layers[0], stride= 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
    def forward(self, g, l):
        mask_0 = torch.zeros(g.shape[0], 128, 1, 1).cuda()
        mask_1 = torch.ones(g.shape[0], 128, 1, 1).cuda()
        mask = torch.cat((mask_0, mask_1), 1).cuda()
        nmask = torch.cat((mask_1, mask_0), 1).cuda()
        f = g * mask + l * nmask
        f = self.remain_bottlenecks(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        return f

    def _remain_bottlenecks(self, block, planes, blocks, stride= 1):
        layers = []
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class GLAMOR_v2(nn.Module):
    def __init__(self, num_classes, loss, block, layers,
                 last_stride = 1, fc_dims = None, dropout_p = None, **kwargs):
        super(GLAMOR_v2, self).__init__()
        self.inplanes = 64
        self.loss = loss
        self.feature_dim = 1024 * block.expansion

        self.GAM = GlobalAttention(self.inplanes)
        self.LAM = LocalAttention(self.inplanes * block.expansion)

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_bottleneck = self._first_bottleneck(block, 64, stride= 1)

        self.branch1 = Branch(self.inplanes, block, layers, last_stride)
        self.branch2 = Branch(self.inplanes, block, layers, last_stride)
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.lnorm = nn.LayerNorm(4096)
        self.fc = self._construct_fc_layer(fc_dims, 1024 * 4, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _first_bottleneck(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer
        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.GAM(x)
        g = self.first_bottleneck(x)
        l = self.LAM(g)

        print(g.shape)
        print(l.shape)

        f = self.branch1(g, l)
        k = self.branch2(l, g)
        
        return torch.cat((f, k), 1)

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        s = self.lnorm(v)
        
        if not self.training:
            return v

        y = self.classifier(s)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri', 'center'}:
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def glamor50_v2(num_classes, loss = {'xent', 'htri', 'center'}, **kwargs):
    model = GLAMOR_v2(
        num_classes = num_classes,
        loss = loss,
        block = Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    return model

def glamor18_v2(num_classes, loss = {'xent', 'htri', 'center'}, **kwargs):
    model = GLAMOR_v2(
        num_classes = num_classes,
        loss = loss,
        block = Bottleneck,
        layers=[2, 2, 2, 2],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    return model