from torchvision import datasets, transforms
from cutout import Cutout
from subpolicy import SubPolicy
from cifarpolicy import CIFAR10Policy
from torch.utils.data import DataLoader

def GetCifar10(batchsize, download):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  #CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=download)
    test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=download)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2)
    return train_dataloader, test_dataloader