from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable

drive_link = "'/content/drive/My Drive/retina-weights'"
training_log = 'training_log.txt'

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batchsize', default = 2, type = int)
parser.add_argument('--classnum', default = 4, type = int)
parser.add_argument('--epochsnum', default = 30, type = int)
parser.add_argument('--datapath', type = str)
parser.add_argument('--trainAnnotations', type = str)
parser.add_argument('--valAnnotations', type = str)





args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

data_path = args.datapath 
train_annotations = args.trainAnnotations
val_annotations = args.valAnnotations
batch_size = args.batchsize
num_classes = args.classnum
epochs_num = args.epochsnum



trainset = ListDataset(root=data_path,
                       list_file=train_annotations, train=True, transform=transform, input_size=600)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=trainset.collate_fn)

testset = ListDataset(root=data_path,
                      list_file=val_annotations, train=False, transform=transform, input_size=600)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size // 2, shuffle=False, num_workers=0, collate_fn=testset.collate_fn)

# Model
net = RetinaNet(num_classes)
net.load_state_dict(torch.load('net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()


criterion = FocalLoss(num_classes)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.item(), train_loss/(batch_idx+1)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.item()
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.item(), test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss
        cmd = "cp './checkpoint/ckpt.pth' " + drive_link
        os.system(cmd)

        with open(training_log, 'a') as f:
          f.write(str(epoch) + "  :  " + str(test_loss) + '\n')

        fcmd = "cp " + training_log + " " + drive_link
        os.system(fcmd)

        

print('training starts with: ' + str(epochs_num) + ' epochs..')

for epoch in range(start_epoch, start_epoch+epochs_num):
    train(epoch)
    test(epoch)


