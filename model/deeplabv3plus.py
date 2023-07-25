from torch import nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from model import Model, Context_Model, Rotation_Model, Jigsaw_Model


class Deeplabv3plus(Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(Deeplabv3plus, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name='resnet50',
            in_channels=input_channel,
            depth=5,
            weights='imagenet',
            output_stride=16,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )

        self.pred_seg = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.initialize(self.decoder, self.pred_seg)


class Context_Deeplabv3plus(Context_Model):
    def __init__(self, args, input_channel=3, criteria=None, inpainting=True):
        super(Context_Deeplabv3plus, self).__init__(
            args=args, criteria=criteria, inpainting=inpainting
        )

        self.encoder = get_encoder(
            name='resnet50',
            in_channels=input_channel,
            depth=5,
            weights=None,
            output_stride=16,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=input_channel, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.initialize(self.decoder, self.pred)


class Rotation_Deeplabv3plus(Rotation_Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(Rotation_Deeplabv3plus, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name=args.encoder, in_channels=input_channel, depth=5, weights=None
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels[-1], 4, bias=True),
        )

        self.initialize(None, self.pred)


class Jigsaw_Deeplabv3plus(Jigsaw_Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(Jigsaw_Deeplabv3plus, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name=args.encoder, in_channels=input_channel, depth=5, weights=None
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
