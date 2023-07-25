# -*- coding: utf-8 -*-
import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from datasets import load_dataset
from model import load_model, load_series_model
from utils.losses import load_criteria


# Train worker: worker for training segmentation task
def train_worker(args):
    kfold = 5 if args.kfold else 1

    for fold in range(kfold):
        train_dataset, val_dataset = load_dataset(args, fold=fold, train=True)
        criteria = load_criteria(args)
        model = load_model(criteria, args)
        epoch = args.epoch
        save_path = os.path.join(args.save_name, str(args.seed))

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        # optionally resume from a checkpoint
        model_path = None
        if args.resume == fold:
            model_path = "{}/last.ckpt".format(args.save_name)
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))

        checkpoint_best = ModelCheckpoint(
            dirpath=save_path,
            monitor="Validation Dice",
            mode="max",
            filename="model_{}_segmentation_best_{}".format(
                str(args.dataset_name), str(fold)
            ),
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            filename="model_{}_segmentation_checkpoint_{}".format(
                str(args.dataset_name), str(fold)
            ),
        )

        trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_best, checkpoint_callback],
            max_epochs=epoch,
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader, ckpt_path=model_path)


# Train aug worker: worker for training MIM and corresponding proxy tasks
def train_aug_worker(args):
    kfold = 5 if args.kfold else 1

    # Defining series model
    for fold in range(kfold):
        train_dataset, val_dataset = load_dataset(
            args,
            fold=fold,
            train=True,
            aug_k=args.aug_k,
            aug_n=args.aug_n,
            patch=args.patch,
        )
        criteria = load_criteria(args)
        models = load_series_model(criteria, args)
        epoch = args.epoch

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=False,
        )

        model_path = None
        if args.resume == fold:
            model_path = "{}/model_inpainting_checkpoint_{}.ckpt".format(
                args.save_name, fold
            )
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))

        # Training for inpainting model
        checkpoint_inpainting_best = ModelCheckpoint(
            dirpath=args.save_name,
            monitor="Inpainting Validation Loss",
            mode="min",
            filename="model_inpainting_best_{}".format(str(fold)),
        )

        checkpoint_inpainting_callback = ModelCheckpoint(
            dirpath=args.save_name,
            filename="model_inpainting_checkpoint_{}".format(str(fold)),
        )

        inpainting_trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_inpainting_best, checkpoint_inpainting_callback],
            max_epochs=epoch,
            log_every_n_steps=10
        )

        print("=> Start training inpainting model")
        inpainting_trainer.fit(models["inpainting"], train_loader, val_loader, ckpt_path=model_path)
        print("=> Inpainting model training finished")

        # Start self supervised validation steps
        # Load best inpainting model
        best_model_path = "{}/model_inpainting_best_{}.ckpt".format(
            args.save_name, str(fold)
        )
        print("=> Loading weights from {}.".format(best_model_path))

        models["rotation"].init_weights(pretrained=best_model_path)
        models["colorization"].init_weights(pretrained=best_model_path)
        models["jigsaw"].init_weights(pretrained=best_model_path)

        for param in models["colorization"].encoder.parameters():
            param.requires_grad = False
        print("=> Loading weight finished")

        # Training steps for rotation model
        checkpoint_rotation_best = ModelCheckpoint(
            dirpath=args.save_name,
            monitor="Rotation Validation Acc",
            mode="max",
            filename="model_rotation_best_{}".format(str(fold)),
        )

        checkpoint_rotation_callback = ModelCheckpoint(
            dirpath=args.save_name,
            filename="model_rotation_checkpoint_{}".format(str(fold)),
        )

        rotation_trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_rotation_best, checkpoint_rotation_callback],
            max_epochs=epoch / 2,
        )

        print("=> Start training rotation model")
        rotation_trainer.fit(models['rotation'], train_loader, val_loader)
        print("=> Rotation model training finished")

        # Training steps for colorization model
        checkpoint_colorization_best = ModelCheckpoint(
            dirpath=args.save_name,
            monitor="Colorization Validation Loss",
            mode="min",
            filename="model_colorization_best_{}".format(str(fold)),
        )

        checkpoint_colorization_callback = ModelCheckpoint(
            dirpath=args.save_name,
            filename="model_colorization_checkpoint_{}".format(str(fold)),
        )

        colorization_trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_colorization_best, checkpoint_colorization_callback],
            max_epochs=epoch,
        )

        print("=> Start training colorization model")
        colorization_trainer.fit(models['colorization'], train_loader, val_loader)
        print("=> Colorization model training finished")

        checkpoint_jigsaw_best = ModelCheckpoint(
            dirpath=args.save_name,
            monitor="Jigsaw Validation Loss",
            mode="min",
            filename="model_jigsaw_best_{}".format(str(fold)),
        )

        checkpoint_jigsaw_callback = ModelCheckpoint(
            dirpath=args.save_name,
            filename="model_jigsaw_checkpoint_{}".format(str(fold)),
        )

        jigsaw_trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_jigsaw_best, checkpoint_jigsaw_callback],
            max_epochs=epoch,
        )

        print("=> Start training jigsaw model")
        jigsaw_trainer.fit(models['jigsaw'], train_loader, val_loader)
        print("=> Jigsaw model training finished")
