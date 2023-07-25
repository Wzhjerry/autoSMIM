import os
from collections import OrderedDict
from importlib import import_module
import imageio
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score
from medpy.metric import dc, jc
from torch.optim.lr_scheduler import CosineAnnealingLR


from utils.train_utils import jigsaw, make_optimizer, rotate_images


class Base_Module(LightningModule):
    def __init__(self, args):
        super(Base_Module, self).__init__()

        self.args = args

    def init_weights(self, pretrained):
        if os.path.isfile(pretrained):
            print("=> loading pretrained model {}".format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            pretrained_dict = pretrained_dict["state_dict"]
            model_dict = self.state_dict()
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                # print('Pretrained dict: ', k)
                if k in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k].shape:
                        available_pretrained_dict[k] = v
                if k[7:] in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k[7:]].shape:
                        available_pretrained_dict[k[7:]] = v

            for k, _ in available_pretrained_dict.items():
                print("loading {}".format(k))
            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict, strict=True)

    def load_weights(self, path):
        if os.path.isfile(path):
            print("=> Loading model from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "module." in k:
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print("=> trained model loaded")

    def initialize(self, decoder, pred):
        if decoder is not None:
            for m in decoder.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        for m in pred.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optim, kwargs_optimizer = make_optimizer(args=self.args)
        opt = optim(self.parameters(), **kwargs_optimizer)
        scheduler = CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]


class Model(Base_Module):
    def __init__(self, args, criteria):
        super(Model, self).__init__(args=args)

        self.criteria = criteria

    def forward(self, x):
        x0 = self.encoder(x)
        x1 = self.decoder(*x0)
        y = self.pred_seg(x1)

        return y

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        if not self.args.dice_loss:
            loss = self.criteria[0](output, target)
        else:
            loss = self.criteria[0](output, target) + 1.5 * self.criteria[1](output, target)
        self.log(
            "Train Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx, unused=None):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def validation_step(self, batch, batch_idx):
        input, target, name = batch

        output = self(input)
        if not self.args.dice_loss:
            loss = self.criteria[0](output, target)
        else:
            loss = self.criteria[0](output, target) + 1.5 * self.criteria[1](output, target)
        dsc = 0
        for j in range(output.size(0)):
            output_temp = torch.argmax(output[j], dim=0).cpu().numpy()
            target_temp = target[j].cpu().numpy()
            dsc += dc(output_temp, target_temp)

        self.log(
            "Validation Loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "Validation Dice",
            dsc / output.size(0),
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )

        return dsc

    def on_test_start(self):
        self.names = []
        self.dscs = []
        self.jacs = []
        self.accs = []
        self.sens = []
        self.HDs = []

    def test_step(self, batch, batch_idx):
        dirname = "{}/results_{}".format(self.args.save_name, self.args.dataset_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        dsc_sum = 0
        jac_sum = 0
        acc_sum = 0

        images, targets_subset, names_subset = batch
        output = self(images)

        for idx, name in enumerate(names_subset):
            output_np = torch.argmax(output[idx], dim=0).cpu().numpy()
            binary_output = np.array(output_np)
            target_np = targets_subset[idx].cpu().numpy().astype(np.uint8)
            if self.args.save_results:
                filename = os.path.join(dirname, str(name) + ".png")
                imageio.imwrite(filename, (binary_output * 255).astype(np.uint8))

            target_1d = np.reshape(target_np, (-1, 1))
            pred_1d = np.reshape(binary_output, (-1, 1))

            accuracy = accuracy_score(target_1d, pred_1d)
            dsc = dc(target_1d, pred_1d)
            jac = jc(target_1d, pred_1d)

            self.names.append(name)
            dsc_sum += dsc
            self.dscs.append(dsc)
            jac_sum += jac
            self.jacs.append(jac)
            acc_sum += accuracy
            self.accs.append(accuracy)

            self.log("Test Dice", dsc, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test Jac", jac, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test Acc", accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_end(self):
        dataframe = pd.DataFrame(
            {
                "name": self.names,
                "dice": self.dscs,
                "jac": self.jacs,
                "acc": self.accs,
            }
        )
        self.args.save_name = os.path.join(self.args.save_name, str(self.args.seed))
        dataframe.to_excel(os.path.join(self.args.save_name, "count_results_{}_{}.xlsx".format(self.args.dataset_name, self.args.fold)))
        results = {
            "Dsc": np.average(self.dscs),
            "Jac": np.average(self.jacs),
            "ACC": np.average(self.accs),
        }
        results_json = json.dumps(results, indent=4)
        with open(os.path.join(self.args.save_name, 'results_{}_{}.json'.format(self.args.dataset_name, self.args.fold)), 'w') as f:
            f.write(results_json)
        print('=> Json result saved')


class Context_Model(Base_Module):
    def __init__(self, args, criteria, inpainting=True):
        super(Context_Model, self).__init__(args=args)

        self.criteria = criteria[0]
        self.inpainting = inpainting

    def forward(self, x):
        input = x[:, :-1]
        mask = x[:, -1:]
        x0 = self.encoder(input)
        x1 = self.decoder(*x0)
        logits = self.pred(x1)
        y = torch.clamp(logits, -1.0, 1.0)
        # mask
        y = y * 0.5 + 0.5
        y *= mask
        y = (y - 0.5) / 0.5

        return y

    def training_step(self, batch, batch_idx):
        if self.inpainting:
            input, _, target = batch
        else:
            _, input, target = batch
        output = self(input)
        loss = self.criteria(output, target)
        loss = loss * 100
        if self.inpainting:
            self.log(
                "Inpainting Train Loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "Colorization Train Loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        if self.inpainting:
            input, _, target, _ = batch
        else:
            _, input, target, _ = batch
        output = self(input)
        loss = self.criteria(output, target)
        loss = loss * 100
        if self.inpainting:
            self.log(
                "Inpainting Validation Loss",
                loss,
                on_epoch=True,
                sync_dist=True,
                on_step=False,
                prog_bar=True,
            )
        else:
            self.log(
                "Colorization Validation Loss",
                loss,
                on_epoch=True,
                sync_dist=True,
                on_step=False,
                prog_bar=True,
            )

        return loss

    def on_test_start(self):
        self.test_loss = []

    def test_step(self, batch, batch_idx):
        images, images_gray, target, names_subset = batch

        if self.inpainting:
            dirname = "{}/results_inpainting".format(self.args.save_name)
            output = self(images)
        else:
            dirname = "{}/results_colorization".format(self.args.save_name)
            output = self(images_gray)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        loss = self.criteria(output, target)

        if self.args.save_results:
            for idx, name in enumerate(names_subset):
                image_np = images.cpu()[idx][:3].permute(1, 2, 0).numpy()
                image_rgb = np.array(image_np) * 0.5 + 0.5
                output_np = output.cpu()[idx].permute(1, 2, 0).numpy()
                output_rgb = np.array(output_np) * 0.5 + 0.5
                filename = os.path.join(dirname, str(name) + ".png")
                imageio.imwrite(
                    filename, ((image_rgb + output_rgb) * 255).astype(np.uint8)
                )

        self.test_loss.append(loss.cpu().numpy())
        self.log("Test Loss", loss)

    def on_test_epoch_end(self):
        norm = np.linalg.norm(self.test_loss)
        norm_loss = np.mean(self.test_loss / norm)
        self.log("Test Norm Loss", norm_loss)


class Rotation_Model(Base_Module):
    def __init__(self, args, criteria):
        super(Rotation_Model, self).__init__(args=args)

        self.criteria = criteria[1]

    def forward(self, x):
        # index = int(x.shape[0] / 4)
        x0 = self.encoder(x)
        y = self.pred(x0[-1])

        return y

    def training_step(self, batch, batch_idx):
        input, _, _ = batch
        input = input[:, :-1, :, :]
        rot_input, rot_target = rotate_images(input)
        output = self(rot_input)

        loss = self.criteria(output, rot_target)
        self.log(
            "Rotation Train Loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input, _, _, _ = batch
        input = input[:, :-1, :, :]
        rot_input, rot_target = rotate_images(input)
        output = self(rot_input)

        loss = self.criteria(output, rot_target)
        acc = accuracy_score(
            torch.argmax(output, dim=1).cpu().numpy(), rot_target.cpu().numpy()
        )

        self.log(
            "Rotation Validation Loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "Rotation Validation Acc",
            acc,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )

    def on_test_start(self):
        self.test_pred = []
        self.test_target = []
        self.test_loss = []

    def test_step(self, batch, batch_idx):
        dirname = "{}/results_rotation".format(self.args.save_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        input, _, _, _ = batch
        input = input[:, :-1, :, :]
        rot_input, rot_target = rotate_images(input)
        output = self(rot_input)

        loss = self.criteria(output, rot_target)
        self.log("Test Loss", loss)

        pred = torch.argmax(output, dim=1)
        self.test_pred.extend(pred.cpu().numpy())
        self.test_target.extend(rot_target.cpu().numpy())
        self.test_loss.append(loss.cpu().numpy())

    def on_test_epoch_end(self):
        norm = np.linalg.norm(self.test_loss)
        norm_loss = np.mean(self.test_loss / norm)
        self.acc = accuracy_score(self.test_target, self.test_pred)
        self.log("Test Acc", self.acc)
        self.log("Test Norm Loss", norm_loss)


class Jigsaw_Model(Base_Module):
    def __init__(self, args, criteria):
        super(Jigsaw_Model, self).__init__(args=args)

        self.criteria = criteria[1]

    def forward(self, x):
        index = int(x.shape[0] / 4)
        for i in range(0, x.shape[0], index):
            piece = x[i:i + index]
            x0 = self.encoder(piece)
            x1 = self.pred_sequential(x0[-1])
            if i == 0:
                pieces = x1
            else:
                pieces = torch.cat([pieces, x1], dim=1)
        y = self.pred_process(pieces)

        return y

    def training_step(self, batch, batch_idx):
        input, _, _ = batch
        input = input[:, :-1, :, :]
        jigsaw_input, jigsaw_target = jigsaw(input, row=2, col=2)
        output = self(jigsaw_input)

        loss = self.criteria(output, jigsaw_target)
        self.log(
            "Jigsaw Train Loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input, _, _, _ = batch
        input = input[:, :-1, :, :]
        jigsaw_input, jigsaw_target = jigsaw(input, row=2, col=2)
        output = self(jigsaw_input)

        loss = self.criteria(output, jigsaw_target)
        acc = accuracy_score(
            torch.argmax(output, dim=1).cpu().numpy(), jigsaw_target.cpu().numpy()
        )

        self.log(
            "Jigsaw Validation Loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "Jigsaw Validation Acc",
            acc,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )

    def on_test_start(self):
        self.test_pred = []
        self.test_target = []
        self.test_loss = []

    def test_step(self, batch, batch_idx):
        dirname = "{}/results_rotation".format(self.args.save_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        input, _, _, _ = batch
        input = input[:, :-1, :, :]
        jigsaw_input, jigsaw_target = jigsaw(input, row=2, col=2)
        output = self(jigsaw_input)

        loss = self.criteria(output, jigsaw_target)
        self.log("Test Loss", loss)

        pred = torch.argmax(output, dim=1)
        self.test_pred.extend(pred.cpu().numpy())
        self.test_target.extend(jigsaw_target.cpu().numpy())
        self.test_loss.append(loss.cpu().numpy())

        # return pred, jigsaw_target, loss

    def on_test_epoch_end(self):
        self.acc = accuracy_score(self.test_target, self.test_pred)
        norm = np.linalg.norm(self.test_loss)
        norm_loss = np.mean(self.test_loss / norm)
        self.log("Test Acc", self.acc)
        self.log("Test Norm Loss", norm_loss)


def load_model(criteria, args):
    print("=> creating model '{}'".format(args.model))
    if "X_Ray" in args.dataset_name:
        input_channel = 1
    else:
        input_channel = 3
    m = import_module("model." + args.model.lower())

    model = getattr(m, args.model)(args, input_channel=input_channel, criteria=criteria)
    if args.pretrained is not None:
        model.init_weights(args.pretrained)

    return model


def load_series_model(criteria, args):
    print("=> creating series model : '{}'".format(args.model))
    if "X_Ray" in args.dataset_name:
        input_channel = 1
    else:
        input_channel = 3
    m = import_module("model." + args.model.lower())
    models = {}

    inpainting_model = "Context_" + args.model
    models["inpainting"] = getattr(m, inpainting_model)(
        args, input_channel=input_channel, criteria=criteria, inpainting=True
    )
    models["colorization"] = getattr(m, inpainting_model)(
        args, input_channel=input_channel, criteria=criteria, inpainting=False
    )

    rot_model = "Rotation_" + args.model
    models["rotation"] = getattr(m, rot_model)(
        args, input_channel=input_channel, criteria=criteria
    )

    rot_model = "Jigsaw_" + args.model
    models["jigsaw"] = getattr(m, rot_model)(
        args, input_channel=input_channel, criteria=criteria
    )

    return models
