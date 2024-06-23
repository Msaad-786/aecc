"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Convolutional Decoder
"""

import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder class.

    Attributes:
        upsample_layers (nn.Sequential): Sequence of upsampling and convolutional layers.
        final_layer (nn.Conv2d): Final convolutional layer to reconstruct the image.
        gate (nn.Module): Gate function for the output.
    """

    def __init__(
        self,
        in_channels=3,
        img_size=224,
        kernel_size=3,
        decoder_channels=(256, 128, 64),
        gate=nn.Sigmoid,
    ) -> None:
        """
        Initialize the ConvDecoder.

        Args:
            in_channels (int, optional): Number of input channels.
            img_size (int, optional): Size of the input image.
            kernel_size (int, optional): Size of the convolution kernels.
            decoder_channels (tuple of int, optional): Number of channels in each upsampling layer.
            gate (nn.Module, optional): Gate function for the output.
        """
        super().__init__()

        layers = []
        input_channels = decoder_channels[0]
        for output_channels in decoder_channels:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                )
            )
            input_channels = output_channels

        self.upsample_layers = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(input_channels, in_channels, kernel_size=kernel_size, padding=1)
        self.gate = gate()

    def forward(self, features):
        """
        Forward pass of the ConvDecoder.

        Args:
            features (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Reconstructed image after the forward pass.
        """
        x = self.upsample_layers(features)
        x = self.final_layer(x)
        x = self.gate(x)
        return x
