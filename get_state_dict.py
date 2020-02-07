'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from fpn import FPN50
from retinanet import RetinaNet
import os

import argparse


parser = argparse.ArgumentParser(description='PyTorch RetinaNet loading model')


parser.add_argument('--classnum', default = 4, type = int)
args = parser.parse_args()
classes_num = args.classnum



# might change it to mobilenet or VGG16


print('Loading pretrained ResNet50 model..')

os.system('wget https://download.pytorch.org/models/resnet50-19c8e357.pth') # downloading the model 

os.rename('resnet50-19c8e357.pth', 'resnet50.pth') # rename 

d = torch.load('resnet50.pth')

print('Loading into FPN50..')
fpn = FPN50()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving RetinaNet..')
net = RetinaNet(num_classes= classes_num)
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), 'net.pth')
print('Done!')

