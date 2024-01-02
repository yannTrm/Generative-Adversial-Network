# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class GeneratorBlock(nn.Module):
    """
    A building block for the Generator model, consisting of a transposed convolutional layer
    followed by optional batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to the input.
        use_batchnorm (bool, optional): Flag to include batch normalization. Default is True.

    Returns:
        torch.Tensor: Output tensor after applying the transposed convolution, batch normalization,
                      and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batchnorm=True):
        super(GeneratorBlock, self).__init__()
        layers = []

        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Generator model for a Generative Adversarial Network (GAN), responsible for generating
    realistic images from random noise.

    Args:
        latent_size (int): Size of the input random noise vector (latent space).
        feature_size (int): Base feature size to be used in the generator architecture.
        num_channels (int): Number of output channels in the generated images.
        img_size (int): Spatial size of the generated images.

    Returns:
        torch.Tensor: Output tensor representing the generated image.

    """

    def __init__(self, latent_size, feature_size, num_channels, img_size):
        super(Generator, self).__init__()

        self.init_size = img_size // 16
        self.l1 = nn.Linear(latent_size, feature_size * 16 * self.init_size**2)

        self.blocks = nn.ModuleList([
            GeneratorBlock(feature_size * 16, feature_size * 8, 4, 2, 1),
            GeneratorBlock(feature_size * 8, feature_size * 4, 4, 2, 1),
            GeneratorBlock(feature_size * 4, feature_size * 2, 4, 2, 1),
            GeneratorBlock(feature_size * 2, num_channels, 4, 2, 1, use_batchnorm=False),
        ])

        self.tanh = nn.Tanh()

    def forward(self, z):
        """
        Forward pass of the Generator.

        Args:
            z (torch.Tensor): Input random noise vector.

        Returns:
            torch.Tensor: Output tensor representing the generated image.

        """
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)

        for block in self.blocks:
            out = block(out)

        img = self.tanh(out)
        return img

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class DiscriminatorBlock(nn.Module):
    """
    A building block for the Discriminator model, consisting of a convolutional layer
    followed by optional batch normalization and LeakyReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to the input.
        use_batchnorm (bool, optional): Flag to include batch normalization. Default is True.

    Returns:
        torch.Tensor: Output tensor after applying the convolution, batch normalization,
                      and LeakyReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batchnorm=True):
        super(DiscriminatorBlock, self).__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    Discriminator model for a Generative Adversarial Network (GAN), responsible for
    distinguishing between real and fake images.

    Args:
        input_size (int): Spatial size of the input images.
        num_channels (int): Number of input channels in the images.
        feature_size (int): Base feature size to be used in the discriminator architecture.

    Returns:
        torch.Tensor: Output tensor representing the discriminator's prediction.

    """

    def __init__(self, input_size, num_channels, feature_size):
        super(Discriminator, self).__init__()

        self.init_size = input_size // 2**4  # Assuming 4 downsampling operations in the discriminator
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(num_channels, feature_size, 4, 2, 1),
            DiscriminatorBlock(feature_size, feature_size * 2, 4, 2, 1),
            DiscriminatorBlock(feature_size * 2, feature_size * 4, 4, 2, 1),
            DiscriminatorBlock(feature_size * 4, feature_size * 8, 4, 2, 1),
        ])

        self.fc = nn.Linear(feature_size * 8 * self.init_size**2, 1)

    def forward(self, x):
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor representing an image.

        Returns:
            torch.Tensor: Output tensor representing the discriminator's prediction.

        """
        for block in self.blocks:
            x = block(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        """
        Forward pass of the GAN.

        Args:
            z (torch.Tensor): Input random noise vector.

        Returns:
            (torch.Tensor, torch.Tensor): Output tensor representing the generated image
                                          and discriminator's prediction for the generated image.

        """
        fake_images = self.generator(z)
        discriminator_output_fake = self.discriminator(fake_images)
        return fake_images, discriminator_output_fake

    def generate_images(self, num_images, latent_size, device):
        """
        Generate images using the generator.

        Args:
            num_images (int): Number of images to generate.
            latent_size (int): Size of the input random noise vector (latent space).
            device (torch.device): Device to which the generated images should be moved.

        Returns:
            torch.Tensor: Generated images.

        """
        z = torch.randn(num_images, latent_size, 1, 1).to(device)
        generated_images = self.generator(z)
        return generated_images
    
    
    def fit1(self, dataloader, num_epochs, latent_size, device):
        """
        Train the GAN.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of training epochs.
            latent_size (int): Size of the input random noise vector (latent space).
            device (torch.device): Device to which the GAN should be moved.

        """
        # Loss function and optimizers
        criterion = nn.BCEWithLogitsLoss()
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(dataloader, 0):
                real_images = real_images.to(device)

                # Training the discriminator
                discriminator_optimizer.zero_grad()
                
                real_labels = torch.ones(real_images.size(0), 1).to(device)
                fake_labels = torch.zeros(real_images.size(0), 1).to(device)

                # Generate fake images
         
                z = torch.randn(real_images.size(0), latent_size).to(device)
                fake_images = self.generator(z)

                # Discriminator loss on real and fake images
                real_loss = criterion(self.discriminator(real_images), real_labels)
                fake_loss = criterion(self.discriminator(fake_images.detach()), fake_labels)
                discriminator_loss = real_loss + fake_loss

                discriminator_loss.backward()
                discriminator_optimizer.step()

                # Training the generator
                generator_optimizer.zero_grad()
                z = torch.randn(real_images.size(0), latent_size).to(device)
                fake_images = self.generator(z)

                # Generator loss
                generator_loss = criterion(self.discriminator(fake_images), real_labels)
                generator_loss.backward()
                generator_optimizer.step()

            # Save generated images at the end of each epoch (optional)
            if epoch % 10 == 0:
                save_image(fake_images, f"image/generated_epoch_{epoch}.png", nrow=int(fake_images.size(0)**0.5), normalize=True)

    def fit(self, dataloader, num_epochs, latent_size, device):
        """
        Train the GAN.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            num_epochs (int): Number of training epochs.
            latent_size (int): Size of the input random noise vector (latent space).
            device (torch.device): Device to which the GAN should be moved.

        """
        # Loss function and optimizers
        criterion = nn.BCEWithLogitsLoss()
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(dataloader, 0):
                real_images = real_images.to(device)

                # Training the discriminator
                discriminator_optimizer.zero_grad()
                
                real_labels = torch.ones(real_images.size(0), 1).to(device)
                fake_labels = torch.zeros(real_images.size(0), 1).to(device)

                # Generate fake images
         
                z = torch.randn(real_images.size(0), latent_size).to(device)
                fake_images = self.generator(z)

                # Discriminator loss on real and fake images
                real_loss = criterion(self.discriminator(real_images), real_labels)
                fake_loss = criterion(self.discriminator(fake_images.detach()), fake_labels)
                discriminator_loss = real_loss + fake_loss

                discriminator_loss.backward()
                discriminator_optimizer.step()

                # Training the generator
                generator_optimizer.zero_grad()
                z = torch.randn(real_images.size(0), latent_size).to(device)
                fake_images = self.generator(z)

                # Generator loss
                generator_loss = criterion(self.discriminator(fake_images), real_labels)
                generator_loss.backward()
                generator_optimizer.step()
                
            print(f"[Epoch {epoch}/{num_epochs}]")


            # Save generated images at the end of each epoch (optional)
            if epoch % 10 == 0:
                save_image(fake_images, f"image/generated_epoch_{epoch}.png", nrow=int(fake_images.size(0)**0.5), normalize=True)



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------