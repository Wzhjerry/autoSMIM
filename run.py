""" Wrapper to train and test a medical image segmentation model. """
import argparse
import os
from argparse import Namespace
import torch
import nni
import yaml
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from lightning.fabric.utilities.seed import seed_everything

import wandb
from utils.train_utils import get_rank
from workers.test_net import test_aug_worker, test_worker
from workers.train_net import train_aug_worker, train_worker

torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="2D Medical Image Segmentation")

    parser.add_argument(
        "--cfg",
        default="./configs.yaml",
        type=str,
        help="Config file used for experiment",
    )
    parser.add_argument(
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    # training configuration
    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--test_batch_size",
        default=12,
        type=int,
        metavar="N",
        help="inference mini-batch size (default: 1)",
    )
    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        metavar="N",
        help="training epoch (default: 100)",
    )
    parser.add_argument(
        "--resume",
        default=-1,
        type=int,
        metavar="N",
        help="resume from which fold (default: -1)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--optimizer",
        default="ADAM",
        choices=("SGD", "ADAM", "RMSprop"),
        help="optimizer to use (SGD | ADAM | RMSprop)",
    )
    parser.add_argument("--pretrained", default="", help="pretrained model weights")
    parser.add_argument("--size", type=int, default=512, help="size of input image")
    parser.add_argument(
        "--dice_loss", action="store_true", help="using dice loss or not"
    )

    # model specifications
    parser.add_argument("--model", type=str, default="UNet", help="model name")
    parser.add_argument(
        "--encoder", type=str, default="resnet50", help="encoder name of the model"
    )
    parser.add_argument(
        "--no_crop", action="store_true", help="disable random resized cropping"
    )

    # experiment configuration
    parser.add_argument(
        "--save_results", action="store_true", help="save context results or not"
    )
    parser.add_argument("--save_name", default="smoke", help="experiment name")
    parser.add_argument(
        "--dataset_name",
        default="staining134",
        help="dataset name [staining134 / dataset / HRF]",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="rotation",
        help="evaluation metric [inpainting / rotation / colirization]",
    )
    parser.add_argument("--kfold", action="store_true", help="5-fold cross-validation")
    parser.add_argument("--smoke_test", action="store_true", help="debug mode")
    parser.add_argument(
        "--description", default="", type=str, help="description of the experiment"
    )
    parser.add_argument(
        "--aug_k", type=int, default=40, help="number of generating superpixels"
    )
    parser.add_argument(
        "--aug_n",
        type=int,
        default=1,
        help="number of superpixel selected for inpainting",
    )
    parser.add_argument("--patch", action="store_true", help="mask using random patch")
    parser.add_argument(
        "--seed", type=int, default=436, help="global setting for random seed"
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=100,
        help="percentage of training data used for training",
    )

    # Wandb configuration
    parser.add_argument(
        "--log_name", default="", type=str, help="description of the wandb logger"
    )
    parser.add_argument("--tags", default=[], help="tags for wandb")
    parser.add_argument(
        "--resume_wandb", action="store_true", help="resume experiment for wandb"
    )
    parser.add_argument(
        "--id", type=str, default="wzh is hangua", help="resume id for wandb"
    )

    parser.add_argument(
        "--update_config",
        action="store_true",
        help="update wandb config for existing experiments",
    )
    parser.add_argument(
        "--optimization", action="store_true", help="doing nni optimization"
    )
    parser.add_argument("--evaluate", action="store_true", help="evaluate only")
    parser.add_argument(
        "--inpainting",
        action="store_true",
        help="doing inpainting and other self supervised progress",
    )

    args = parser.parse_args()

    with open(args.cfg, encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        opt = vars(args)
        opt.update(config)
        args = Namespace(**opt)

    for arg in vars(args):
        if vars(args)[arg] == "True":
            vars(args)[arg] = True
        elif vars(args)[arg] == "False":
            vars(args)[arg] = False

    dirname = "{}".format(args.save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if not args.smoke_test and not args.update_config:
        if args.resume_wandb:
            print("=> Resuming Wandb logger")
            # Need modification if you want to use WandB
            args.logger = WandbLogger(
                project="your_project_name",
                entity="your_entity_name",
                name=args.log_name,
                tags=args.tags,
                resume=True,
                id=args.id,
                notes=args.description,
            )
            if get_rank() == 0:
                args.logger.experiment.config.update(args, allow_val_change=True)
                wandb.run.log_code(".")
        else:
            print("=> Making Wandb logger")
            args.logger = WandbLogger(
                project="your_project_name",
                entity="your_entity_name",
                name=args.log_name,
                tags=args.tags,
                notes=args.description,
            )
            if get_rank() == 0:
                args.logger.experiment.config.update(args, allow_val_change=True)
                wandb.run.log_code(".")
    else:
        print("=> Using local csv logger")
        args.logger = CSVLogger(
            save_dir=args.save_name,
            name=args.log_name,
            flush_logs_every_n_steps=50
        )

    seed_everything(args.seed, workers=True)
    if args.inpainting:
        if args.evaluate:
            test_aug_worker(args, aug_k=args.aug_k, aug_n=args.aug_n)
        else:
            print("=> Start inpainting and self supervised evaluation")
            train_aug_worker(args)
            print("=> Process finished")
    elif args.optimization:
        print("=> Running nni for optimization")
        params = nni.get_next_parameter()
        print("=> Param: ", params)
        nni_result = test_aug_worker(args, **params)
        nni.report_final_result(nni_result)
    elif args.evaluate:
        print("=> Only evaluating")
        test_worker(args)
    else:
        print("=> Start segmentation training process")
        train_worker(args)
        print("=> Segmentation training process finished")

        print("=> Start testing segmentation model")
        test_worker(args)
        print("=> Segmentation model test finished")


if __name__ == "__main__":
    main()
