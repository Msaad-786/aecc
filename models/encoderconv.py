"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Convolutional Encoder
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder class.

    Attributes:
        conv_layers (nn.Sequential): Sequence of convolutional layers.
        flatten (nn.Flatten): Layer to flatten the feature map.
    """

    def __init__(
        self,
        in_channels=3,
        img_size=224,
        kernel_size=3,
        encoder_channels=(64, 128, 256),
    ) -> None:
        """
        Initialize the ConvEncoder.

        Args:
            in_channels (int, optional): Number of input channels.
            img_size (int, optional): Size of the input image.
            kernel_size (int, optional): Size of the convolution kernels.
            encoder_channels (tuple of int, optional): Number of channels in each convolutional layer.
        """
        super().__init__()

        layers = []
        input_channels = in_channels
        for output_channels in encoder_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )
            input_channels = output_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, img):
        """
        Forward pass of the ConvEncoder.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Features after the forward pass.
        """
        features = self.conv_layers(img)
        features = self.flatten(features)
        return features
