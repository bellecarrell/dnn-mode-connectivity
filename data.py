import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from os.path import join as pjoin
import os.path as path

class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    if dataset == 'cifar5m':
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        preproc =     transforms.Compose([
                transforms.ToTensor(),
                normalize]) # numpy unit8 --> [-1, 1] tensor

        X_tr, Y_tr, X_te, Y_te = load_cifar5m(path)
        X_tr, Y_tr = X_tr[45000:], Y_tr[45000:]
        #todo: add preprocessing for data aug once added data aug
        train_set = TransformingTensorDataset(X_tr, Y_tr, transform=preproc)
        
        test_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)

        if use_test:
            ds = getattr(torchvision.datasets, 'CIFAR10')
            path = os.path.join(path, dataset.lower())
            transform = getattr(getattr(Transforms, 'CIFAR10'), transform_name)

            print('You are going to run models on the test set. Are you sure?')
            test_set = ds(path, train=False, download=True, transform=transform.test)

        ys = Y_tr
            
    else: 
        ds = getattr(torchvision.datasets, dataset)
        path = os.path.join(path, dataset.lower())
        transform = getattr(getattr(Transforms, dataset), transform_name)
        train_set = ds(path, train=True, download=True, transform=transform.train)

        if use_test:
            print('You are going to run models on the test set. Are you sure?')
            test_set = ds(path, train=False, download=True, transform=transform.test)
        else:
            print("Using train (45000) + validation (5000)")
            train_set.data = train_set.data[:-5000]
            train_set.targets = train_set.targets[:-5000]

            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.data = test_set.data[-5000:]
            test_set.targets = test_set.targets[-5000:]
        ys = train_set.targets

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, max(ys) + 1

class TransformingTensorDataset(Dataset):
    """TensorDataset with support of torchvision transforms.
    """
    def __init__(self, X, Y, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.X = X
        self.Y = Y
        self.transform = transform
        self.train_labels = self.Y

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        y = self.Y[index]

        return x, y

    def __len__(self):
        return len(self.X)

def download_dir(gpath, localroot='~/tmp/data', no_clobber=True):
    ''' Downloads GCS dir into localdir (if not exists), and returns the local dir path.'''
    import subprocess
    import pickle
    localroot = path.expanduser(localroot)

    nc = '-n' if no_clobber else ''
    subprocess.call(f'mkdir -p {localroot}', shell=True)
    subprocess.call(f'gsutil -m cp {nc} -r {gpath} {localroot}', shell=True)
    localdir = pjoin(localroot, path.basename(gpath))
    return localdir


def load_cifar5m(path=''):
    '''
        Returns 5million synthetic samples.
        warning: returns as numpy array of unit8s, not torch tensors.
    '''

    # todo add flag for nte? keeping at 5k to match other dataset curve training
    nte = 5000 # num. of test samples to use (max 1e6)
    print('Downloading CIFAR 5mil...')
    local_dir = download_dir('gs://gresearch/cifar5m',localroot=path) # download all 6 dataset files
    #local_dir = '/expanse/lustre/projects/csd697/nmallina/data/cifar-5m'

    npart = 1000448
    X_tr = np.empty((5*npart, 32, 32, 3), dtype=np.uint8)
    Ys = []
    print('Loading CIFAR 5mil...')
    for i in range(5):
        z = np.load(pjoin(local_dir, f'part{i}.npz'))
        X_tr[i*npart: (i+1)*npart] = z['X']
        Ys.append(torch.tensor(z['Y']).long())
        print(f'Loaded part {i+1}/6')
    Y_tr = torch.cat(Ys)

    z = np.load(pjoin(local_dir, 'part5.npz')) # use the 6th million for test.
    print(f'Loaded part 6/6')

    X_te = z['X'][:nte]
    Y_te = torch.tensor(z['Y'][:nte]).long()

    return X_tr, Y_tr, X_te, Y_te
