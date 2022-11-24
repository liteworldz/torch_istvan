import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision, torch
import torchvision.transforms as transforms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import torchattacks
import torchvision.models as models
from utils import progress_bar
from datat_set import AugmentedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS= 8/255
ALPHA= 2/255
STEPS= 10

def load_data(ds_name, val_size=5000):
    # Data
    print('==> Preparing data..', ds_name)

    if ds_name == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        ds_klass = torchvision.datasets.CIFAR10
        num_classes = 10
    elif ds_name == "cifar100":

        ds_klass = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise AssertionError('Given Dataset is not supported!')
    trainset = ds_klass(root='./data', train=True, download=True)
    testset = ds_klass(root='./data', train=False, download=True)
    if val_size > 0:
        trainset, valset = torch.utils.data.random_split(trainset, [50000 - val_size, val_size],
                                                         torch.Generator().manual_seed(42))
    else:
        valset = testset
    return trainset, valset, testset, num_classes


def train_loop(net, attack, trainloader, device, optimizer, criterion):
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


def evaluate(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    return acc


def get_model_by_name(name, n_classes):
    if name == "vgg19":
        net = models.resnet34()#VGG('VGG19', num_classes=n_classes)
    elif name == "resnet18":
        net = models.resnet18() #(num_classes=n_classes)
    elif name == "preactresnet18":
        net = models.resnet34()#(num_classes=n_classes)
    elif name == "lenet":
        net = models.resnet34()#LeNet(num_classes=n_classes)
    else:
        raise Exception('Unknown network name: {0}'.format(name))
    # @TODO: add other networks
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    return net


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainset, valset, testset, n_classes = load_data(args.dataset, val_size=args.valsize)
    # Model
    print('==> Building model..')
    net = get_model_by_name(args.netname, n_classes)

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
        print('<< running on cuda >>')
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/adv_' + args.netname + '.pth')
        net.load_state_dict(checkpoint['net'])
        acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    if args.dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "cifar100":
        mean = (129.3 / 255, 124.1 / 255, 112.4 / 255)
        std = (68.2 / 255, 65.4 / 255, 70.4 / 255)
    else:
        raise AssertionError('Given Dataset is not supported!')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # @TODO: add your augmentation here
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainloader_aug = torch.utils.data.DataLoader(AugmentedDataset(trainset, transforms=transform_train),
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=2)
    trainloader = torch.utils.data.DataLoader(AugmentedDataset(trainset, transforms=transform_test),
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(AugmentedDataset(valset, transforms=transform_test),
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(AugmentedDataset(testset, transforms=transform_test),
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('\nEpoch: %d' % epoch)
        train_loop(net, attack, trainloader_aug, device, optimizer, criterion)
        acc = evaluate(net, valloader, device, criterion)
        # Save checkpoint.
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.netname + '.pth')
        scheduler.step()
    train_acc = evaluate(net, trainloader, device, criterion)
    val_acc = evaluate(net, valloader, device, criterion)
    test_acc = evaluate(net, testloader, device, criterion)
    print('train-acc:', train_acc, 'val-acc:', val_acc, 'test-acc:', test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--valsize', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--netname', type=str, default="resnet18")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    main(args)
