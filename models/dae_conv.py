"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Denoising Autoencoder Convolutional (DAE-Conv) models.

The configurations follow the same principles as standard convolutional autoencoders for image denoising.

"""

import torch
import torch.nn as nn

from utils import RayleighChannel

from .decoderconv import ConvDecoder
from .encoderconv import ConvEncoder


import torch
import torch.nn as nn

from utils import RayleighChannel

class CAEConv(nn.Module):
    """
    CAE is a Denoising AutoEncoder model that leverages the Convolutional architecture.
    It consists of an encoder and a decoder, both of which are based on convolutional layers.
    """

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
        """
        Initializes the CAE model with the given parameters.

        Args:
            in_channels (int, optional): The number of channels in the input image.
            img_size (int, optional): The size (height and width) of the input image in pixels.
            kernel_size (int, optional): The size of the convolution kernels.
            encoder_channels (tuple of int, optional): The number of channels in each convolutional layer of the encoder.
            decoder_channels (tuple of int, optional): The number of channels in each upsampling layer of the decoder.
            gate (nn.Module, optional): The gate function used in the decoder.
            noise_factor (float, optional): The factor by which the input image is noised before being passed to the encoder.
        """
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
        """
        Forward pass of the CAE.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Predicted image after the forward pass.
        """
        features = self.encoder(img)
        noisy_features = self.rayleigh(features)
        
        # Calculate the correct size for reshaping
        batch_size = img.size(0)
        channels = self.encoder_channels[-1]
        feat_size = self.img_size // (2 ** len(self.encoder_channels))  # Assuming each conv layer halves the image size

        # Check if the size is correct
        expected_elements = batch_size * channels * feat_size * feat_size
        if noisy_features.numel() != expected_elements:
            raise RuntimeError(f"Shape mismatch: {noisy_features.numel()} elements cannot be reshaped to {[batch_size, channels, feat_size, feat_size]}")

        noisy_features = noisy_features.view(batch_size, channels, feat_size, feat_size)
        predicted_img = self.decoder(noisy_features)

        return predicted_img


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
