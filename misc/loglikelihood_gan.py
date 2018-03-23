import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time
from itertools import *


# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# test_dataset = dsets.MNIST(root='./data/',
#                            train=False,
#                            transform=transforms.ToTensor())

# parameters
batch_size = 64
z_dim = 64
X_height = train_dataset.train_data.shape[1] # 28
X_width = train_dataset.train_data.shape[2] # 28
lr = 5e-5
beta1 = 0.5
beta2 = 0.999


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

def log(x):
    return torch.log(x + 1e-10)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (X_height // 4) * (X_width // 4)),
            nn.BatchNorm1d(128 * (X_height // 4) * (X_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128,  (X_height // 4),  (X_width // 4))
        x = self.deconv(x)
        return x


class Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(128 *  (X_height // 4) *  (X_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
            nn.Tanh(),
        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(128 *  (X_height // 4) *  (X_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (X_height // 4) * (X_width // 4))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 *  (X_height // 4) *  (X_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
            nn.Linear(z_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (X_height // 4) * (X_width // 4))
        x = self.fc(x)
        return x

E = Encoder()
G = Generator()
D = Discriminator()

# cuda
if torch.cuda.is_available():
    E.cuda()
    G.cuda()
    D.cuda()

def reset_grad():
    E.zero_grad()
    G.zero_grad()
    D.zero_grad()

# Define solvers
D_solver = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
G_solver = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
E_solver = optim.Adam(E.parameters(), lr=lr, betas=(beta1, beta2))

epoch = 40
for ep in range(1,epoch+1):
    print("Epoch {} started!".format(ep))
    # epoch_start_time = time.time()
    for iter, (X, _) in enumerate(train_loader):
        # Sample z and data
        X = to_var(X)

        """Discriminator"""
        z = to_var(torch.randn(batch_size, z_dim))
        X_hat = G(z)
        D_real = D(X)
        D_fake = D(X_hat)
        D_loss = -torch.mean(log(D_real) + log(1 - D_fake))
        # Optimize
        D_loss.backward()
        D_solver.step()
        reset_grad()

        """Generator"""
        z = to_var(torch.randn(batch_size, z_dim))
        X_hat = G(z)
        D_fake = D(X_hat)
        G_loss = -torch.mean(log(D_fake))
        # Optimize
        G_loss.backward()
        G_solver.step()
        reset_grad()

        """Encoder"""
        z = to_var(torch.randn(batch_size, z_dim))
        X_hat = G(z)
        z_mu, z_sigma = E(X_hat)
        E_loss = torch.mean(torch.sum((z - z_mu)**2 * torch.exp(-z_sigma) + z_sigma, 1)) # - loglikehood
        # Optimize
        E_loss.backward()
        E_solver.step()
        reset_grad()

        """ Plot """
        if (iter+1) == train_loader.dataset.__len__() // batch_size:
            # Print and plot every epoch
            print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}\n'
                  .format(ep, D_loss.data[0], G_loss.data[0], E_loss.data[0]))

            # Reconstruction
            mu, sigma = E(X)
            X_hat = G(z)
            eps = to_var(torch.randn(batch_size, z_dim))
            X_rec = G(mu + eps * torch.exp(sigma/2.0))
            if torch.cuda.is_available():
                samples = X_hat.cpu().data.numpy().transpose(0, 2, 3, 1) # 1
                origins = X.cpu().data.numpy().transpose(0, 2, 3, 1) # 2
                recons = X_rec.cpu().data.numpy().transpose(0, 2, 3, 1) # 3
            else:
                samples = X_hat.data.numpy().transpose(0, 2, 3, 1)
                origins = X.data.numpy().transpose(0, 2, 3, 1) # 2
                recons = X_rec.data.numpy().transpose(0, 2, 3, 1) # 3

            # Save images
            save_images(origins[:4 * 4, :, :, :], [4, 4],
                          '/output/original' + '_epoch%03d' % ep + '.png')
            save_images(samples[:4 * 4, :, :, :], [4, 4],
                          '/output/random' + '_epoch%03d' % ep + '.png')
            save_images(recons[:4 * 4, :, :, :], [4, 4],
                          '/output/reconstructed' + '_epoch%03d' % ep + '.png')

            break
