# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import torch.nn as nn

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, latent_size, generator_feature_size, image_channels):
        """
        Generator Class
        
        Parameters:
        - latent_size (int): Size of the latent vector (i.e., size of the generator input).
        - generator_feature_size (int): Size of feature maps in the generator.
        - image_channels (int): Number of channels in the training images. For color images, this is 3.
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, generator_feature_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_feature_size * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_size * 16, generator_feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_feature_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_size * 8, generator_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_feature_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_size * 4, generator_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_feature_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_size * 2, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        Forward pass of the generator.
        
        Parameters:
        - input: Input tensor.
        
        Returns:
        - Tensor: Generated output.
        """
        return self.main(input)

#------------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, image_channels, discriminator_feature_size):
        """
        Discriminator Class
        
        Parameters:
        - image_channels (int): Number of channels in the training images. For color images, this is 3.
        - discriminator_feature_size (int): Size of feature maps in the discriminator.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, discriminator_feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_size, discriminator_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_size * 2, discriminator_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_size * 4, discriminator_feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Forward pass of the discriminator.
        
        Parameters:
        - input: Input tensor.
        
        Returns:
        - Tensor: Output prediction.
        """
        return self.main(input)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------