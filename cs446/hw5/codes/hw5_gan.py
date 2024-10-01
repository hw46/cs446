import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw5_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layer
        self.fc1 = nn.Linear(8*5*5, 1)

        # Initialize weights
        self._weight_init()

    def _weight_init(self):
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        # Layer 1: Convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Layer 2: Convolution
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Layer 3: Convolution
        x = self.conv3(x)
        x = F.relu(x)

        # Flatten the output for the dense layer
        x = torch.flatten(x, start_dim=1)

        # Linear layer
        x = self.fc1(x)
        return x


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()
        
        # Initialize the layers as per the provided architecture
        self.linear = nn.Linear(zdim, 1568, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # No bias, used to increase spatial dimensions
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # No bias, used to increase spatial dimensions
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self._weight_init()

    def _weight_init(self):
        
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # Pass through the linear layer
        x = self.linear(z)
        x = self.leaky_relu(x)
        
        # Reshape the output to match the required dimensions for convolution
        x = x.view(-1, 32, 7, 7)  # Reshape to (batch_size, channels, height, width)
        
        # Apply the first upsampling and convolution
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        
        # Apply the second upsampling and convolution
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        
        # Apply the final convolution and sigmoid activation
        x = self.conv3(x)
        x = self.sigmoid(x)
        
        return x


class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)
        
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        
        # Real data part
        real_data = batch_data.to(self._dev)
        real_labels = torch.ones(batch_size, 1, device=self._dev)  # Real labels = 1
        real_predictions = self.disc(real_data)
        loss_real = self.bce_logits_loss(real_predictions, real_labels)
        
        # Fake data part
        fake_data = self.gen(z).detach()  # Detach to stop gradients to generator
        fake_labels = torch.zeros(batch_size, 1, device=self._dev)  # Fake labels = 0
        fake_predictions = self.disc(fake_data)
        loss_fake = self.bce_logits_loss(fake_predictions, fake_labels)

        # Total loss is the sum of the loss for real and fake
        loss_d = (loss_real + loss_fake) / 2
        return loss_d

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.

        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        
        # Generate fake data
        fake_data = self.gen(z)

        # We want the discriminator to believe that this fake data is real.
        # Therefore, target labels are all ones (real)
        real_labels = torch.ones(batch_size, 1, device=self._dev)

        # Calculate discriminator's response to fake data
        fake_predictions = self.disc(fake_data)

        # Calculate loss
        loss_g = self.bce_logits_loss(fake_predictions, real_labels)
        return loss_g

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()