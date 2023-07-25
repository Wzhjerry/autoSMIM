import os
import math
import shutil
import torch
import torch.optim as optim


def save_checkpoint(
    state, is_best, fold, savename, epoch, filename="model_checkpoint.pth.tar"
):
    dirname = "{}".format(savename)
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        print("Saving checkpoint {} as the best model...".format(epoch))
        shutil.copyfile(
            os.path.join(dirname, filename),
            "{}/model_best_{}.pth.tar".format(savename, str(fold)),
        )


def adjust_learning_rate(optimizer, epoch, epochs, lr, cos=True, schedule=None):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_alpha(epoch, epochs, min_alpha=0.2):
    step = (1 - min_alpha) / epochs
    return 1 - epoch * step


def make_optimizer(args):
    """make optimizer"""
    # optimizer
    kwargs_optimizer = {"lr": args.lr}

    if args.optimizer == "SGD":
        optimizer_class = optim.SGD
        kwargs_optimizer["momentum"] = 0.9
    elif args.optimizer == "ADAM":
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8
    elif args.optimizer == "RMSprop":
        optimizer_class = optim.RMSprop
        kwargs_optimizer["eps"] = args.epsilon
    else:
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8

    return optimizer_class, kwargs_optimizer


def rotate_images(images):
    nimages = images.shape[0]
    n_rot_images = 4 * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros(
        [n_rot_images, images.shape[1], images.shape[2], images.shape[3]]
    ).to(images.device)
    rot_classes = torch.zeros([n_rot_images]).long().to(images.device)

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages : 2 * nimages] = images.flip(3).transpose(2, 3)
    rot_classes[nimages : 2 * nimages] = 1
    # rotate 180
    rotated_images[2 * nimages : 3 * nimages] = images.flip(3).flip(2)
    rot_classes[2 * nimages : 3 * nimages] = 2
    # rotate 270
    rotated_images[3 * nimages : 4 * nimages] = images.transpose(2, 3).flip(3)
    rot_classes[3 * nimages : 4 * nimages] = 3

    return rotated_images, rot_classes


def jigsaw(images, row, col):
    label_list = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ]

    size_hori = int(images.shape[2] / row)
    size_ver = int(images.shape[3] / col)

    puzzle_images = torch.zeros(
        [images.shape[0] * 4, images.shape[1], size_hori, size_ver]
    ).to(images.device)
    puzzle_classes = torch.zeros([images.shape[0]]).long().to(images.device)

    for cnt in range(images.shape[0]):
        temp_img = images[cnt]
        label = torch.randint(high=24, size=(1,)).long().to(images.device)
        for i in range(4):
            num = label_list[label][i]
            div = num // col
            mod = num % col
            puzzle_images[i * images.shape[0] + cnt] = temp_img[
                :,
                div * size_hori:(div + 1) * size_hori,
                mod * size_ver:(mod + 1) * size_ver,
            ]

        puzzle_classes[cnt] = label

    return puzzle_images, puzzle_classes


def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor
