import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from augmentation import RandAugment
from _02_custom_dataset import CustomDataset 

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
#normal_std = (0.5, 0.5, 0.5)
normal_std = (0.25, 0.25, 0.25)
    
def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)
# base_dataset.targets  <- list임
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split_test(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_custom(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(args.resize,args.resize)),  # resize 부분 추가 
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(args.resize,args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = CustomDataset(args.csv_train_filename,args.data_path, train=True)
# base_dataset.targets  <- list임
    # num_labeled 값 설정(train data의 length 대비하여 x_u_split 함수  호출 전에 설정). num_classes의 배수
    #args.num_labeled = args.num_classes * (len(base_dataset.targets) // args.num_classes)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split_test(args, base_dataset.targets)
    train_labeled_dataset = CustomSSL(
        args.csv_train_filename,
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
#     print("data here!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print(transform_labeled)
    train_unlabeled_dataset = CustomSSL(
        args.csv_train_filename,
        args.data_path, train_unlabeled_idxs,
        train=True,
        # 20220114 수정
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
        # transform=TransformMPL(args, mean=normal_mean, std=normal_std)
    )
    test_dataset = CustomDataset(args.csv_test_filename,args.data_path, train=False,
                                    transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



def x_u_split(args, labels):
    #label_per_class = args.num_labeled // args.num_classes  # 10개 class arg에서 최초 label된 것은 4000개임
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)  # class당 400개 label
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))  # 0~50000 숫자 array
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]  # i번째 class와 일치하는 인덱스의 array
        idx = np.random.choice(idx, label_per_class, True)  # idx array에서 랜덤으로 400개 추출
#         idx = np.random.choice(idx, label_per_class, False)  # idx array에서 랜덤으로 400개 추출
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    
    assert len(labeled_idx) == args.num_labeled  # 4000개 labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx  # labeled idx는 추출작업을 했지만 unlabeled_idx는 추출작업 없음


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.Resize(size=(args.resize,args.resize)), # resize 때문에 추가
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect')])
        self.aug = transforms.Compose([
            transforms.Resize(size=(args.resize,args.resize)), # resize 때문에 추가
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),  # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor              
            # 2022.01.14 주석처리
            # transforms.Normalize(mean=mean, std=std)
            ])  #Normalize a tensor image with mean and standard deviation.

    def __call__(self, x):
        ori = self.ori(x)  # weak augmentation 에 해당??
        aug = self.aug(x)  # strong augmentation  에 해당 
        return self.normalize(ori), self.normalize(aug)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

import os 
from skimage import io

class CustomSSL(CustomDataset):
    def __init__(self, csv_file,root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(csv_file,root, train=train,
                         transform=transform,
                         target_transform=target_transform)

        if indexs is not None:
            self.data = self.data.iloc[indexs]  #csv를 읽은 df에서 0번째 컬럼
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img_path=os.path.join(self.root_dir, str(self.data.iloc[index,0]))
        target=self.targets[index]
        # ---- 변경 코드 시작-----------
        img = io.imread(img_path, pilmode='RGB')
        # ---- 변경 코드 끝-----------
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'custom': get_custom}
