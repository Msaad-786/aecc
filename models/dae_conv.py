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


class CAEConv(nn.Module):
    """
    CAEConv is a Denoising AutoEncoder convmodel that leverages Convolutional Neural Network (CNN)
    architecture. It consists of an encoder and a decoder, both of which are based on CNNs.

    Attributes:
        encoder (ConvEncoder): The encoder part of the autoencoder. It is responsible for
        transforming the input image into a lower-dimensional representation.
        decoder (ConvDecoder): The decoder part of the autoencoder. It reconstructs the original image from the lower-dimensional representation produced by the encoder.

    The CAEConv convmodel is initialized with several parameters that define the structure and behavior
    of the encoder and decoder. These parameters include the number of input channels, the size of
    the input image, the size of the kernels, the number of convolutional layers and channels in the
    encoder and decoder, the gate function for the decoder, and the noise factor for the input image.
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
        Initializes the CAEConv convmodel with the given parameters.

        Args:
            in_channels (int, optional): The number of channels in the input image.
            img_size (int, optional): The size (height and width) of the input image in pixels.
            kernel_size (int, optional): The size of the kernels used in the convolutions.
            encoder_channels (tuple of int, optional): The number of channels in each convolutional layer in the encoder.
            decoder_channels (tuple of int, optional): The number of channels in each convolutional layer in the decoder.
            gate (nn.Module, optional): The gate function used in the decoder.
            noise_factor (float, optional): The factor by which the input image is noised before
            being passed to the encoder.
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

    def forward(self, img):
        """
        Forward pass of the CAEConv.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Predicted image after the forward pass.
        """
        features = self.encoder(img)
        noisy_features = self.rayleigh(features)
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
