import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision, torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from models import *
#import torchvision.models as models
from utils import progress_bar
import dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks


EPS= 8/255
ALPHA= 2/255
STEPS= 10

def train_loop(net, trainloader, device, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        

def adv_train_loop(net, attack, trainloader, device, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        adv = attack(inputs, targets)
        outputs = net(adv)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# .to(device)

# Evaluate results on clean data
def evaluate(net, testloader, device, criterion):
    print("Evaluating single model results on clean data")
    total = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for xs, ys in testloader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            preds1 = net(xs)
            preds_np1 = preds1.cpu().detach().numpy()
            finalPred = np.argmax(preds_np1, axis=1)
            correct += (finalPred == ys.cpu().detach().numpy()).sum()
            total += len(xs)
    acc = float(correct) / total
    #print('Clean accuracy: %.2f%%' % (acc * 100))
    return acc * 100


# Evaluate results on adversarially perturbed
def robustness(net, attack, testloader, device, criterion):
    print("Evaluating single model results on adv data")
    total = 0
    correct = 0
    net.eval()
    for xs, ys in testloader:
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        xs, ys = Variable(xs), Variable(ys)
        adv = attack(xs, ys)
        preds1 = net(adv)
        preds_np1 = preds1.cpu().detach().numpy()
        finalPred = np.argmax(preds_np1, axis=1)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += testloader.batch_size
    acc = float(correct) / total
    #print('Adv accuracy: {:.3f}％'.format(acc * 100))
    return acc * 100


def get_model_by_name(name, n_classes):
    if name == "vgg19":
        model = ResNet18() #models.resnet18()
    elif name == "resnet18":
        model = ResNet18() # models.resnet18()
    else:
        raise Exception('Unknown network name: {0}'.format(name))
    return model


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, valloader, testloader = dataset.get_loader(args.val_size, args.batch_size)
    # Model
    print('==> Building model..')
    net = get_model_by_name(args.netname, 10)

    # @TODO: find out what is the used intialization?
    # default sems work quite well, but he_uniform might be better?
    # def initialize(m):
    #     if type(m) == nn.Linear:
    #         torch.nn.init.xavier_uniform(m.weight)
    # torch.nn.init.xavier_normal_(m.weight)
    # m.bias.data.fill_(0)

    # net.apply(initialize)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.netname + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))['net']
        net.load_state_dict(checkpoint)
        acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('\nEpoch: %d' % epoch)
        if args.adv:
            adv_train_loop(net, attack, trainloader, device, optimizer, criterion)
        else:
            train_loop(net, trainloader, device, optimizer, criterion)
        acc = evaluate(net, valloader, device, criterion)
        # Save checkpoint.
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' +
                   args.netname)
        scheduler.step()
    train_acc = evaluate(net, trainloader, device, criterion)
    val_acc = evaluate(net, valloader, device, criterion)
    test_acc = evaluate(net, testloader, device, criterion)
    robust_test_acc = robustness(net, attack, testloader, device, criterion)
    print('train-acc:', train_acc, 'val-acc:', val_acc, 'test-acc:', test_acc, 'robust_test-acc:', robust_test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--val_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--netname', type=str, default="resnet18")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--adv', '-a', action='store_true', help='adversarial training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    main(args)
