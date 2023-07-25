import random
import glob
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets.data_utils import analyze_name, random_crop


cv2.setNumThreads(1)


class ISIC16(Dataset):
    def __init__(self, x, y, names, im_transform, label_transform, train=False):
        self.im_transform = im_transform
        self.label_transform = label_transform
        assert len(x) == len(y)
        assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.names = names
        self.train = train

    def __len__(self):
        if self.train:
            return self.dataset_size * 2
        else:
            return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
            return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._get_index(idx)

        # BGR -> RGB -> PIL
        _input = cv2.imread(self.x[idx])[..., ::-1]
        _input = cv2.resize(_input, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # label
        _target = cv2.imread(self.y[idx])
        _target = cv2.resize(_target, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        _target = _target[..., 0]

        name = self.names[idx]
        mask = np.ones_like(_target)

        if self.train:
            image, label = random_crop(_input, _target, roi=mask, size=[0.6, 1.0])
        else:
            image = _input.copy()
            label = _target.copy()

        im = Image.fromarray(np.uint8(image))
        target = Image.fromarray(np.uint8(label)).convert("1")

        # identical transformation for im and gt
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            im_t = self.im_transform(im)

        torch.manual_seed(seed)
        random.seed(seed)
        if self.label_transform is not None:
            target_t = self.label_transform(target)
            target_t = torch.squeeze(target_t).long()

        # For debugging
        # import imageio
        # im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t.numpy()) * 255
        # imageio.imwrite('./debug/im.png', np.array(im_np).astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', np.array(target_np).astype(np.uint8))

        if self.train:
            return im_t, target_t
        else:
            return im_t, target_t, name


def load_name():

    inputs, targets, names = [], [], []
    test_inputs, test_targets, test_names = [], [], []

    # Need modification
    input_pattern = glob.glob(
        "/data/share/wangzh/datasets/ISIC2016/Training_Data/*.jpg"
    )
    targetlist = (
        "/data/share/wangzh/datasets/ISIC2016/Training_GroundTruth/{}_Segmentation.png"
    )

    input_pattern.sort()

    for i in tqdm(range(len(input_pattern))):
        inputpath = input_pattern[i]
        name = analyze_name(inputpath)
        targetpath = targetlist.format(str(name))

        if os.path.exists(inputpath):
            inputs.append(inputpath)
            targets.append(targetpath)
            names.append(name)

    inputs = np.array(inputs)
    targets = np.array(targets)
    names = np.array(names)

    # Need modification
    test_input_pattern = glob.glob(
        "/data/share/wangzh/datasets/ISIC2016/Test_Data/*.jpg"
    )
    test_targetlist = (
        "/data/share/wangzh/datasets/ISIC2016/Test_GroundTruth/{}_Segmentation.png"
    )

    test_input_pattern.sort()

    for j in tqdm(range(len(test_input_pattern))):
        test_inputpath = test_input_pattern[j]
        test_name = analyze_name(test_inputpath)
        test_targetpath = test_targetlist.format(str(test_name))

        if os.path.exists(test_inputpath):
            test_inputs.append(test_inputpath)
            test_targets.append(test_targetpath)
            test_names.append(test_name)

    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)
    test_names = np.array(test_names)

    assert len(test_inputs) == len(test_targets)
    assert len(test_targets) == len(test_names)

    return inputs, targets, names, test_inputs, test_targets, test_names


def load_dataset(args, fold, train=True, aug_k=40, aug_n=1, patch=False):
    inputs, targets, names, test_inputs, test_targets, test_names = load_name()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    kf = KFold(n_splits=5, shuffle=True, random_state=726)
    for ifold, (train_index, val_index) in enumerate(kf.split(inputs)):
        if ifold != fold:
            continue
        X_train, X_val = inputs[train_index], inputs[val_index]
        y_train, y_val = targets[train_index], targets[val_index]
        names_train, names_val = names[train_index], names[val_index]
    X_test = test_inputs
    y_test = test_targets
    names_test = test_names

    # transform input images and construct datasets
    size = args.size
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(size),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.0, 0.0),
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

    train_dataset = ISIC16(
        X_train,
        y_train,
        names_train,
        im_transform=transform,
        label_transform=label_transform,
        train=True,
    )
    val_dataset = ISIC16(
        X_val,
        y_val,
        names_val,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
    )
    test_dataset = ISIC16(
        X_test,
        y_test,
        names_test,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
    )

    if train:
        return train_dataset, val_dataset
    else:
        return test_dataset
