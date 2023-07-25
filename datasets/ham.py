import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from datasets.data_utils import analyze_name, random_crop

cv2.setNumThreads(1)


class Ham(Dataset):
    def __init__(
        self,
        x,
        y,
        names,
        im_transform,
        label_transform,
        train=False,
        aug_k=40,
        aug_n=1,
    ):
        self.im_transform = im_transform
        self.label_transform = label_transform
        assert len(x) == len(names)
        self.dataset_size = len(x)
        self.x = x
        self.y = y
        self.names = names
        self.train = train
        self.aug_k = aug_k
        self.aug_n = aug_n

    def __len__(self):
        if self.train:
            return self.dataset_size
        else:
            return self.dataset_size

    def __getitem__(self, idx):

        # raw image & label
        input = cv2.imread(self.x[idx])[..., ::-1]
        input = cv2.resize(input, (512, 512), interpolation=cv2.INTER_CUBIC)
        if os.path.exists(self.y[idx]):
            target = np.load(self.y[idx]).astype(np.uint16)
        else:
            target = slic(input, n_segments=self.aug_k, compactness=5, start_label=1)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)

        name = self.names[idx]
        mask = np.ones_like(target)

        if self.train:
            image, rawlabel = random_crop(input, target, roi=mask, size=[0.2, 0.8])
        else:
            image = input.copy()
            rawlabel = target.copy()

        # select a superpixel
        label = np.zeros_like(rawlabel, dtype=np.uint8)
        sp_idx_pool = np.random.permutation(np.max(rawlabel)) + 1
        sp_idx_pool = sp_idx_pool[: self.aug_n]
        sp_idx_pool.sort()

        for sp_idx in sp_idx_pool:
            label[np.where(rawlabel == sp_idx)] = 1

        image_c = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = Image.fromarray(np.uint8(image))
        image_c = np.concatenate([np.expand_dims(image_c, -1)] * 3, -1)
        image_c = Image.fromarray(np.uint8(image_c))
        label = Image.fromarray(np.uint8(label * 255)).convert("1")

        # identical transformation for im and gt
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            image_t = self.im_transform(image)
            torch.manual_seed(seed)
            random.seed(seed)
            image_c = self.im_transform(image_c)
            torch.manual_seed(seed)
            random.seed(seed)
            label_t = self.label_transform(label)

            target_t2 = image_t * 0.5 + 0.5
            target_t2 *= label_t
            target_t2 = (target_t2 - 0.5) / 0.5
            target_t3 = image_t * 0.5 + 0.5
            target_t4 = image_c * 0.5 + 0.5
            target_t3 *= 1 - label_t
            target_t4 *= label_t
            target_t4 = target_t4 + target_t3
            target_t3 = (target_t3 - 0.5) / 0.5
            target_t4 = (target_t4 - 0.5) / 0.5

        # For debugging
        # import imageio
        # im_np = (target_t3.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # temp = (target_t3.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        # im_np1 = mark_boundaries(temp, target, color=(1, 1, 1)) * 255
        # im_np2 = mark_boundaries(temp, target, color=(1, 1, 0)) * 255
        # im_np3 = mark_boundaries(temp, target, color=(0, 1, 1)) * 255
        # im_np4 = mark_boundaries(temp, target, color=(1, 0, 0)) * 255
        # im_c_np = (target_t4.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t2.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # label_np = label_t.permute(1, 2, 0).numpy() * 255
        # imageio.imwrite('./debug/im.png', np.array(im_np).astype(np.uint8))
        # imageio.imwrite('./debug/im1.png', np.array(im_np1).astype(np.uint8))
        # imageio.imwrite('./debug/im2.png', np.array(im_np2).astype(np.uint8))
        # imageio.imwrite('./debug/im3.png', np.array(im_np3).astype(np.uint8))
        # imageio.imwrite('./debug/im4.png', np.array(im_np4).astype(np.uint8))
        # imageio.imwrite('./debug/im_c.png', np.array(im_c_np).astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', np.array(target_np).astype(np.uint8))
        # imageio.imwrite('./debug/label.png', np.array(label_np).astype(np.uint8))

        im_t = torch.cat([target_t3, label_t], dim=0)
        im_c = torch.cat([target_t4, label_t], dim=0)

        if self.train:
            return im_t, im_c, target_t2
        else:
            return im_t, im_c, target_t2, name


def load_name(args, aug_k):

    inputs, targets, names = [], [], []
    input_list = glob.glob("/data/share/wangzh/datasets/Ham10000/part1/*.jpg")
    target_pattern = "/data/share/wangzh/datasets/Ham10000/T_sp_{}/{}.npy"
    input_list.sort()

    for inputpath in tqdm(input_list):
        name = analyze_name(inputpath)
        targetpath = target_pattern.format(aug_k, name)

        if os.path.exists(inputpath):
            inputs.append(inputpath)
            targets.append(targetpath)
            names.append(name)

    input_list = glob.glob("/data/share/wangzh/datasets/Ham10000/part2/*.jpg")
    input_list.sort()

    for inputpath in tqdm(input_list):
        name = analyze_name(inputpath)
        targetpath = target_pattern.format(aug_k, name)

        if os.path.exists(inputpath):
            inputs.append(inputpath)
            targets.append(targetpath)
            names.append(name)

    inputs = np.array(inputs)
    targets = np.array(targets)
    names = np.array(names)

    return inputs, targets, names


def load_dataset(args, fold, train=True, aug_k=40, aug_n=1, patch=False):

    inputs, targets, names = load_name(args, aug_k)
    index = int(args.percent * len(inputs) / 100)
    inputs = inputs[:index]
    targets = targets[:index]
    names = names[:index]
    print("Length of new inputs:", len(inputs))
    # mean & variance
    # mean, std = stat(inputs, masks)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=726)
    for ifold, (train_index, test_index) in enumerate(kf.split(inputs)):
        if ifold != fold:
            continue
        X_trainset, X_test = inputs[train_index], inputs[test_index]
        y_trainset, y_test = targets[train_index], targets[test_index]
        names_trainset, names_test = names[train_index], names[test_index]

        # transform input images and construct datasets
        size = args.size
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(size),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.05, 0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
        label_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(size, interpolation=Image.NEAREST),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_label_transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

        train_dataset = Ham(
            X_trainset,
            y_trainset,
            names_trainset,
            im_transform=transform,
            label_transform=label_transform,
            train=True,
            aug_k=aug_k,
            aug_n=aug_n,
        )
        val_dataset = Ham(
            X_test,
            y_test,
            names_test,
            im_transform=test_transform,
            label_transform=test_label_transform,
            train=False,
            aug_k=aug_k,
            aug_n=aug_n,
        )

    if train:
        return train_dataset, val_dataset
    else:
        return val_dataset
