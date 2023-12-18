from enum import Enum
import json

import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from data_proc.augmentation import ColourDistortion
from data_proc.dataset import *
from resnet import *

from torchvision.datasets.vision import VisionDataset
from typing import Optional, Callable, Tuple, Any
import os
import pickle
import numpy as np
import os.path
from PIL import Image


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        # if not check_integrity(path, self.meta["md5"]):
        #     raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class SupportedDatasets(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TINY_IMAGENET = "tiny_imagenet"
    IMAGENET = "imagenet"
    STL10 = "stl10"

Datasets = namedtuple('Datasets', 'trainset testset clftrainset num_classes stem')

def get_datasets(dataset: str, augment_clf_train=False, add_indices_to_data=False, num_positive=2):

    CACHED_MEAN_STD = {
        SupportedDatasets.CIFAR10.value: ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        SupportedDatasets.CIFAR100.value: ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        SupportedDatasets.STL10.value: ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        SupportedDatasets.TINY_IMAGENET.value: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        SupportedDatasets.IMAGENET.value: ((0.485, 0.456, 0.3868), (0.2309, 0.2262, 0.2237))
    }

    PATHS = {
        SupportedDatasets.CIFAR10.value: '/data/cifar10/',
        SupportedDatasets.CIFAR100.value: '/data/cifar100/',
        SupportedDatasets.STL10.value: '/data/stl10/',
        SupportedDatasets.TINY_IMAGENET.value: '/data/tiny_imagenet/',
        SupportedDatasets.IMAGENET.value: '/data/ILSVRC/'
    }

    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]

    # Data
    if dataset == SupportedDatasets.STL10.value:
        img_size = 96
    elif dataset == SupportedDatasets.IMAGENET.value:
        img_size = 224
    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
        img_size = 64
    else:
        img_size = 32

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ColourDistortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    if dataset == SupportedDatasets.IMAGENET.value:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])

    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_clftrain = transform_test
    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_clftrain = transform_test

    trainset = testset = clftrainset = num_classes = stem = None
    
    if dataset == SupportedDatasets.CIFAR100.value:
        if add_indices_to_data:
            dset = add_indices(CIFAR100)    # use modified CIFAR100 instead of torchvision.datasets.CIFAR100
        else:
            dset = CIFAR100    # use modified CIFAR100 instead of torchvision.datasets.CIFAR100
            trainset = CIFAR100Augment(root=root, train=True, download=False, transform=transform_train, n_augmentations=num_positive)
        clftrainset = dset(root=root, train=True, download=False, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=False, transform=transform_test)
        num_classes = 100
        stem = StemCIFAR

    elif dataset == SupportedDatasets.CIFAR10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10 
            trainset = CIFAR10Augment(root=root, train=False, download=True, transform=transform_train, n_augmentations=num_positive)
        clftrainset = dset(root=root, train=True, download=False, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=False, transform=transform_test)
        num_classes = 10
        stem = StemCIFAR
    elif dataset == SupportedDatasets.STL10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STL10
            trainset = STL10Augment(root=root, split='train+unlabeled', download=True, transform=transform_train)
        clftrainset = dset(root=root, split='train', download=True, transform=transform_clftrain)
        testset = dset(root=root, split='test', download=True, transform=transform_test)
        num_classes = 10
        stem = StemSTL

    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
        if add_indices_to_data:
            raise NotImplementedError("Not implemented for TinyImageNet")
        trainset = ImageFolderAugment(root=f"{root}train/", transform=transform_train, n_augmentations=num_positive)  
        clftrainset = ImageFolder(root=f"{root}train/", transform=transform_clftrain)      
        testset = ImageFolder(root=f"{root}test/", transform=transform_train)    
        num_classes = 200
        stem = StemCIFAR
    
    elif dataset == SupportedDatasets.IMAGENET.value:
        if add_indices_to_data:
            raise NotImplementedError("Not implemented for ImageNet")
        trainset = ImageNetAugment(root=f"{root}train_full/", transform=transform_train, n_augmentations=num_positive)
        clftrainset = ImageNet(root=f"{root}train_full/", transform=transform_clftrain)      
        testset = ImageNet(root=f"{root}test/", transform=transform_clftrain)     
        num_classes = 1000
        stem = StemImageNet

    return Datasets(trainset=trainset, testset=testset, clftrainset=clftrainset, num_classes=num_classes, stem=stem)
