import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import ImageFolder


import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from typing import Optional, Callable, Tuple, Any
import os
import pickle
import os.path


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
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
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
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR10Augment(CIFAR10):
    def __init__(self, root: str, transform=Callable, n_augmentations: int = 2, train: bool = True, download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            List of augmented views of element at index
        """
        img = self.data[index]
        pil_img = Image.fromarray(img)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs

class STL10Augment(torchvision.datasets.STL10):
    def __init__(
            self,
            root: str,
            split: str,
            transform: Callable,
            n_augmentations: int = 2,
            download: bool = False,
        ) -> None:
            super().__init__(
                root=root,
                split=split,
                transform=transform,
                download=download)
            self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs
    

class CIFAR100Augment(CIFAR10Augment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class ImageFolderAugment(ImageFolder):
    def __init__(self, root: str, transform=Callable, n_augmentations: int = 2):
        super().__init__(
            root=root,
            transform=transform,
        )
        self.n_augmentations = n_augmentations
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        pil_img = self.loader(path)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs

def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
        

class ImageNetAugment(torch.utils.data.Dataset):
    def __init__(self, root, transform, n_augmentations=2):
        self.root = root
        self.transform = transform 
        self.n_augmentations = n_augmentations
        df = pd.read_csv(os.path.join(root, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pil_img = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs