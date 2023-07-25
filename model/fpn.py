from torch import nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from model import Model, Context_Model


class FPN(Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(FPN, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name="resnet50", in_channels=input_channel, depth=5, weights=None
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            merge_policy="add",
        )

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=self.decoder.out_channels, out_channels=2, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.initialize(decoder=self.decoder, pred=self.pred)

    def forward(self, x):
        x0 = self.encoder(x)
        x1 = self.decoder(*x0)
        y = self.pred(x1)

        return y


class Context_FPN(Context_Model):
    def __init__(self, args, input_channel=3, criteria=None, inpainting=True):
        super(Context_FPN, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name="resnet50", in_channels=input_channel, depth=5, weights=None
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            merge_policy="add",
        )

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=self.decoder.out_channels, out_channels=3, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.initialize(decoder=self.decoder, pred=self.pred)
