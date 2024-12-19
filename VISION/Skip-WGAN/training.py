from utilities import gradient_penalty, weight_initialization, CustomDataset, get_data_loaders
from model import Generator, Discriminator
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
means = torch.tensor([0.5, 0.5, 0.5]).to(device)
stds = torch.tensor([0.5, 0.5, 0.5]).to(device)
BATCH_SIZE = 32

transform_aug = transforms.Compose(
    [
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
    #transforms.RandomAffine(degrees=(30, 180), translate=(0.2, 0.2), scale=(0.5, 1.1)),
    transforms.RandomHorizontalFlip(0.55),
    transforms.RandomVerticalFlip(0.55),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
    transforms.RandomAutocontrast(p=0.333)
    ]
)

path = ""
dataset = CustomDataset(root_dir=path, transform=transform_aug)
channels, img, img_dim, train_loader = get_data_loaders(dataset, BATCH_SIZE)


# Instantiate models
gen = Generator(128, 3, 1).to(device)   
disc = Discriminator(3).to(device)

# Initialize weights
weight_initialization(gen)
weight_initialization(disc)

disc_optim = optim.Adam(disc.parameters(), lr = 2e-4, betas=(0.0, 0.9))
gen_optim = optim.Adam(gen.parameters(), lr = 5e-5, betas=(0.0, 0.9))
gen.train()
disc.train()


generated_images = []
epochs = 100
LAMBDA_GP = 10.0
latent_dim = 128

# training loop 
for epoch in range(epochs):
    epoch_start_time = time.time()
    for batch_idx, (real_img) in enumerate(train_loader):
        real_img = real_img.to(device)
        x = 5
        if torch.rand(1) > 0.6:
            x = 6
        for _ in range(x):
            noise = (torch.randn(BATCH_SIZE, latent_dim, 1, 1) + (torch.randn(BATCH_SIZE, latent_dim, 1, 1) * 0.01)).to(device)
            gen_img = gen(noise)
            disc_real_img = disc(real_img).view(-1)
            disc_gen_img = disc(gen_img).view(-1)
            grad_penalty = gradient_penalty(disc,real_img,gen_img,device)
            loss_disc = -(torch.mean(disc_real_img) - torch.mean(disc_gen_img)) + (LAMBDA_GP * grad_penalty)
            disc.zero_grad() 
            loss_disc.backward(retain_graph=True)
            disc_optim.step() 

        output = disc(gen_img).view(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()


    with torch.no_grad():
        generated = (gen(noise)).reshape([*img.shape]).to(device)
        generated = (generated * torch.tensor(stds).view(3, 1, 1)) + torch.tensor(means).view(3, 1, 1).to(device)
        generated_images.append(generated)
        real = real_img.reshape([*img.shape])


    print(f"At epoch {epoch + 1}, the duration was {((time.time() - epoch_start_time)/60):.2f} minutes!")
    print(f"Discriminator Loss: {loss_disc:.4f}, Generator loss: {loss_gen:.4f}")

    x = generated_images[-1][0].detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    plt.imshow(x)
    plt.show()

    generator_states = gen.state_dict()
    discriminator_states = disc.state_dict()
    torch.save(generator_states, "generator.pth")
    torch.save(discriminator_states,"discriminator.pth")
    print("Saved!")