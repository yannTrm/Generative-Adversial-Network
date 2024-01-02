# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from load_dataset import CustomDataset
from model.DCGAN import Generator, Discriminator


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def train_gan(generator, discriminator, dataloader, optimizerG, optimizerD, BCELoss, class_label, num_epochs, path_save_image, save_path_generator, save_path_discriminator):
    print(f"Starting Training Loop for class {class_label}...")
    
    for e in range(num_epochs):
        running_loss_D = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data
            real_labels = torch.ones(len(inputs), 1, 1, 1, dtype=torch.float, device=device)
            fake_labels = torch.zeros(len(inputs), 1, 1, 1, dtype=torch.float, device=device)
    
            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()
            predicted_real = discriminator(inputs.to(device))
            loss_real = BCELoss(predicted_real, real_labels)
            loss_real.backward()    
    
            noise = torch.randn(len(inputs), latent_size, 1, 1, device=device)
            fake_image = generator(noise)
            predicted_fake = discriminator(fake_image.detach())
            loss_fake = BCELoss(predicted_fake, fake_labels)
            loss_fake.backward()
            loss_Discriminator = loss_real + loss_fake
            optimizerD.step()
    
            # Update Generator: maximize log(D(G(z)))
            generator.zero_grad()
            predicted_fake = discriminator(fake_image)
            loss_gen = BCELoss(predicted_fake, real_labels)  # Using real labels for generator loss
            loss_gen.backward()
            optimizerG.step()
            running_loss_D += loss_Discriminator.item() / len(dataloader)
    
        # Plotting and saving generated images
        noise = torch.randn(1, latent_size, 1, 1, device=device)
        fake_img = generator(noise)
        save_image(fake_img, f'{path_save_image}/FakeIMG_{class_label}_{e}.png')
    
        print(f"Epoch {e} \n Completion: {e/num_epochs*100} percent")
    
    print(f'Finished Training for class {class_label}')

    torch.save(generator.state_dict(), save_path_generator)
    torch.save(discriminator.state_dict(), save_path_discriminator)






#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__=="__main__":
    root_dir = '../data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs = 200
    latent_size = 10
    feature_size = 64
    num_channels = 3
    
    img_size = 64
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


    dataset = CustomDataset(root_dir, mode='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    indices_class_0 = [i for i, (_, label) in enumerate(dataset) if label == 0] 
    subset_class_0 = Subset(dataset, indices_class_0)
    dataloader_class_0 = DataLoader(subset_class_0, batch_size=1, shuffle=True, num_workers=4)
    
    indices_class_1 = [i for i, (_, label) in enumerate(dataset) if label == 1] 
    subset_class_1 = Subset(dataset, indices_class_1)
    dataloader_class_1 = DataLoader(subset_class_1, batch_size=1, shuffle=True, num_workers=4)
    
    
    
    generator_class_0 = Generator(latent_size, feature_size, num_channels).to(device)
    discriminator_class_0 = Discriminator(num_channels, feature_size).to(device)
    
    generator_class_1 = Generator(latent_size, feature_size, num_channels).to(device)
    discriminator_class_1 = Discriminator(num_channels, feature_size).to(device)
    

    
    optimizerG0 = optim.Adam(generator_class_0.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    optimizerD0 = optim.Adam(discriminator_class_0.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    BCELoss0 = nn.BCELoss()
    
    optimizerG1 = optim.Adam(generator_class_1.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    optimizerD1 = optim.Adam(discriminator_class_1.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    BCELoss1 = nn.BCELoss()
    


    # Entraînement pour chaque classe
    train_gan(generator_class_0, discriminator_class_0,
              dataloader_class_0, optimizerG0,
              optimizerD0, BCELoss0, 
              class_label=0, num_epochs=num_epochs, 
              path_save_image='./img/dcgan/class0', 
              save_path_generator='./model/dcgan/generator_model_class_0.pth',
              save_path_discriminator='./model/dcgan/discriminator_model_class_0.pth')
    
    """
    train_gan(generator_class_1, discriminator_class_1,
              dataloader_class_1, optimizerG1,
              optimizerD1, BCELoss1,
              class_label=1, num_epochs=num_epochs,
              path_save_image='./img/dcgan/class1',
              save_path_generator='./model/dcgan/generator_model_class_1.pth',
              save_path_discriminator='./model/dcgan/discriminator_model_class_1.pth')
    """

