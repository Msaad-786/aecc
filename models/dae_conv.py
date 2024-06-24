import torch
import torch.nn as nn
from utils import RayleighChannel
from .decoderconv import ConvDecoder
from .encoderconv import ConvEncoder

class CAEConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        img_size=224,
        kernel_size=3,
        encoder_channels=(64, 128, 256),
        decoder_channels=(256, 128, 64),
        gate=nn.Sigmoid,
        noise_factor=0.2,
    ) -> None:
        super().__init__()

        self.encoder = ConvEncoder(
            in_channels,
            img_size,
            kernel_size,
            encoder_channels,
        )
        self.decoder = ConvDecoder(
            in_channels,
            img_size,
            kernel_size,
            decoder_channels,
            gate,
        )

        self.rayleigh = RayleighChannel(noise_factor)

        self.img_size = img_size
        self.encoder_channels = encoder_channels

    def forward(self, img):
        features = self.encoder(img)
        
        # Debug: Check encoder output shape
        print(f"Encoder output shape: {features.shape}")
        
        noisy_features = self.rayleigh(features)
        
        # Calculate the correct size for reshaping
        batch_size = img.size(0)
        channels = self.encoder_channels[-1]
        feat_size = self.img_size // (2 ** len(self.encoder_channels))  # Assuming each conv layer halves the image size

        # Debug information
        print(f"Noisy features shape: {noisy_features.shape}")
        print(f"Expected shape: {[batch_size, channels, feat_size, feat_size]}")

        # Check if the size is correct
        expected_elements = batch_size * channels * feat_size * feat_size
        if noisy_features.numel() != expected_elements:
            raise RuntimeError(f"Shape mismatch: {noisy_features.numel()} elements cannot be reshaped to {[batch_size, channels, feat_size, feat_size]}")

        noisy_features = noisy_features.view(batch_size, channels, feat_size, feat_size)
        predicted_img = self.decoder(noisy_features)

        return predicted_img

# Example model instantiation
def dae_conv_tiny(**kwargs):
    convmodel = CAEConv(
        encoder_channels=(32, 64, 128),
        decoder_channels=(128, 64, 32),
        **kwargs
    )
    return convmodel

def dae_conv_small(**kwargs):
    convmodel = CAEConv(
        encoder_channels=(64, 128, 256),
        decoder_channels=(256, 128, 64),
        **kwargs
    )
    return convmodel

def dae_conv_base(**kwargs):
    convmodel = CAEConv(
        encoder_channels=(128, 256, 512),
        decoder_channels=(512, 256, 128),
        **kwargs
    )
    return convmodel

def dae_conv_large(**kwargs):
    convmodel = CAEConv(
        encoder_channels=(256, 512, 1024),
        decoder_channels=(1024, 512, 256),
        **kwargs
    )
    return convmodel

def dae_conv_huge(**kwargs):
    convmodel = CAEConv(
        encoder_channels=(512, 1024, 2048),
        decoder_channels=(2048, 1024, 512),
        **kwargs
    )
    return convmodel

dae_conv_models = {
    "dae_conv_tiny": dae_conv_tiny,
    "dae_conv_small": dae_conv_small,
    "dae_conv_base": dae_conv_base,
    "dae_conv_large": dae_conv_large,
    "dae_conv_huge": dae_conv_huge,
}
