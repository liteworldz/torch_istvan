import torch
import torchvision
import torchvision.transforms as transforms

MEAN = [0.4914, 0.4822, 0.4465]
#STD  = [0.2471, 0.2435, 0.2616]
STD  = [0.2023, 0.1994, 0.2010]

def get_loader(val_size, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    #val_size = 5000
    trainset, valset = torch.utils.data.random_split(trainset, [50000 - val_size, val_size],
                                                         torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=4)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size, shuffle=False, num_workers=4)
    
    return trainloader, valloader, testloader
