import glob
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from datasets.data_utils import analyze_name

cv2.setNumThreads(1)


class ISIC17(Dataset):
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
        return self.dataset_size

    def __getitem__(self, idx):
        # image
        input = cv2.imread(self.x[idx])[..., ::-1]
        input = cv2.resize(input, (512, 512), interpolation=cv2.INTER_CUBIC)
        # label
        target = cv2.imread(self.y[idx])
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
        target = target[..., 0]
        # name
        name = self.names[idx]

        im = Image.fromarray(np.uint8(input))
        target = Image.fromarray(np.uint8(target)).convert("1")

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
    val_inputs, val_targets, val_names = [], [], []
    test_inputs, test_targets, test_names = [], [], []

    # Need modification
    input_pattern = glob.glob(
        "/data/share/wangzh/datasets/ISIC2017/Training_Data/*.jpg"
    )
    targetlist = (
        "/data/share/wangzh/datasets/ISIC2017/Training_GroundTruth/{}_segmentation.png"
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
    val_input_pattern = glob.glob(
        "/data/share/wangzh/datasets/ISIC2017/Validation_Data/*.jpg"
    )
    val_targetlist = "/data/share/wangzh/datasets/ISIC2017/Validation_GroundTruth/{}_segmentation.png"

    val_input_pattern.sort()

    for j in tqdm(range(len(val_input_pattern))):
        val_inputpath = val_input_pattern[j]
        val_name = analyze_name(val_inputpath)
        val_targetpath = val_targetlist.format(str(val_name))

        if os.path.exists(val_inputpath):
            val_inputs.append(val_inputpath)
            val_targets.append(val_targetpath)
            val_names.append(val_name)

    val_inputs = np.array(val_inputs)
    val_targets = np.array(val_targets)
    val_names = np.array(val_names)

    # Need modification
    test_input_pattern = glob.glob(
        "/data/share/wangzh/datasets/ISIC2017/Test_Data/*.jpg"
    )
    test_targetlist = (
        "/data/share/wangzh/datasets/ISIC2017/Test_GroundTruth/{}_segmentation.png"
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

    return (
        inputs,
        targets,
        names,
        val_inputs,
        val_targets,
        val_names,
        test_inputs,
        test_targets,
        test_names,
    )


def load_dataset(args, fold, train=True, aug_k=40, aug_n=1):
    (
        inputs,
        targets,
        names,
        val_inputs,
        val_targets,
        val_names,
        test_inputs,
        test_targets,
        test_names,
    ) = load_name()

    index = int(args.percent * len(inputs) / 100)
    if args.percent < 1:
        inputs = inputs[447:index + 449]
        targets = targets[447:index + 449]
        names = names[447:index + 449]
    else:
        inputs = inputs[:index]
        targets = targets[:index]
        names = names[:index]
    print("Length of new inputs:", len(inputs))

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    X_trainset, X_test = inputs, test_inputs
    y_trainset, y_test = targets, test_targets
    train_names_set, names_test = names, test_names

    X_train, X_val, y_train, y_val, names_train, names_val = (
        X_trainset,
        val_inputs,
        y_trainset,
        val_targets,
        train_names_set,
        val_names,
    )

    # transform input images and construct datasets
    size = args.size
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ColorJitter(0.2, 0.2, 0.0, 0.0),
            transforms.ToTensor(),
            normalize,
        ]
    )
    label_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.NEAREST),
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

    train_dataset = ISIC17(
        X_train,
        y_train,
        names_train,
        im_transform=transform,
        label_transform=label_transform,
        train=True,
    )
    val_dataset = ISIC17(
        X_val,
        y_val,
        names_val,
        im_transform=test_transform,
        label_transform=test_label_transform,
        train=False,
    )
    test_dataset = ISIC17(
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
