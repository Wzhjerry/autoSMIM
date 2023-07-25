import torch.nn as nn

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from model import Model, Context_Model, Rotation_Model, Jigsaw_Model


class EfficientNetv2(Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(EfficientNetv2, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name='tu-tf_efficientnetv2_s',
            in_channels=3,
            depth=5,
            weights='imagenet',
            output_stride=32
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )

        self.pred_seg = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1),
        )

        self.initialize(decoder=self.decoder, pred=self.pred_seg)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(*x)
        y = self.pred_seg(x)
        return y


class Context_EfficientNetv2(Context_Model):
    def __init__(self, args, input_channel=3, criteria=None, inpainting=True):
        super(Context_EfficientNetv2, self).__init__(args=args, criteria=criteria, inpainting=inpainting)

        self.encoder = get_encoder(
            name='tu-tf_efficientnetv2_s',
            in_channels=input_channel,
            depth=5,
            weights='imagenet',
            output_stride=32
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
        )

        self.initialize(decoder=self.decoder, pred=self.pred)


class Rotation_EfficientNetv2(Rotation_Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(Rotation_EfficientNetv2, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name='tu-tf_efficientnetv2_s',
            in_channels=input_channel,
            depth=5,
            weights='imagenet',
            output_stride=32
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels[-1], 4, bias=True),
        )

        self.initialize(None, self.pred)


class Jigsaw_EfficientNetv2(Jigsaw_Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(Jigsaw_EfficientNetv2, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name='tu-tf_efficientnetv2_s',
            in_channels=input_channel,
            depth=5,
            weights='imagenet',
            output_stride=32
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.pred_sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels[-1], 512, bias=True),
        )

        self.pred_process = nn.Sequential(
            nn.Linear(2048, 24, bias=True),
        )

        self.initialize(None, self.pred_sequential)
        self.initialize(None, self.pred_process)
