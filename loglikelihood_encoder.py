import utils, torch, time, os, pickle
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import scipy.misc
import imageio
import matplotlib.gridspec as gridspec
from itertools import *

"""Parameters"""
batch_size = 64
z_dim = 64
lr = 2e-5
beta1 = 0.5
beta2 = 0.999
dset = 'cifar10'
epoch = 5

"""Load in dataset"""
if dset == 'mnist':
    train_loader = DataLoader(dataset=datasets.MNIST(root='./data/',
                                                     train=True,
                                                     transform=transforms.ToTensor(),
                                                     download=True),
                              batch_size=batch_size,
                              shuffle=True)
elif dset == 'fashion-mnist':
    train_loader = DataLoader(dataset=datasets.FashionMNIST(root='./data/',
                                                            train=True,
                                                            download=True,
                                                            transform=transforms.Compose([transforms.ToTensor()])),
                              batch_size=batch_size,
                              shuffle=True)
elif dset == 'celebA':
    train_loader = utils.load_celebA('./data/',
                         transform=transforms.Compose([transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]),
                         batch_size=batch_size,
                         shuffle=True)
elif dset == 'cifar10':
    train_loader = DataLoader(dataset=datasets.CIFAR10(root='./data/',
                                                     train=True,
                                                     transform=transforms.ToTensor(),
                                                     download=True),
                              batch_size=batch_size,
                              shuffle=True)

"""Generator"""
class Generator(nn.Module):
    def __init__(self, dataset = 'mnist', z_dim = 64):
        super(Generator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = z_dim # z dim
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = z_dim # z dim
            self.output_dim = 3
        elif dataset == 'cifar10':
            self.input_height = 32
            self.input_width = 32
            self.input_dim = z_dim # z dim
            self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

"""Encoder"""
class Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist', z_dim = 64):
        super(Encoder, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = z_dim
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = z_dim
        elif dataset == 'cifar10':
            self.input_height = 32
            self.input_width = 32
            self.input_dim = 3
            self.output_dim = z_dim

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma

"""Discriminator"""
class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(Discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1
        elif dataset == 'cifar10':
            self.input_height = 32
            self.input_width = 32
            self.input_dim = 3
            self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)

        return x

G = Generator(dataset=dset, z_dim=z_dim)
E = Encoder(dataset=dset, z_dim=z_dim)
D = Discriminator(dataset=dset)

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

"""Training"""
for ep in range(1,epoch+1):
    print("Epoch {} started!".format(ep))
    # epoch_start_time = time.time()
    for iter, (X, _) in enumerate(train_loader):
        # Sample z and data
        X = utils.to_var(X)

        """Discriminator"""
        z = utils.to_var(torch.randn(batch_size, z_dim))
        X_hat = G(z)
        D_real = D(X)
        D_fake = D(X_hat)
        D_loss = -torch.mean(utils.log(D_real) + utils.log(1 - D_fake))
        # Optimize
        D_loss.backward()
        D_solver.step()
        reset_grad()

        """Encoder"""
        z = utils.to_var(torch.randn(batch_size, z_dim))
        X_hat = G(z)
        z_mu, z_sigma = E(X_hat)
        E_loss = torch.mean(torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma, 1))  # - loglikehood
        # Optimize
        E_loss.backward()
        E_solver.step()
        reset_grad()

        """Generator"""
        # Use both Discriminator and Encoder to update Generator
        z = utils.to_var(torch.randn(batch_size, z_dim))
        X_hat = G(z)
        D_fake = D(X_hat)
        z_mu, z_sigma = E(X_hat)
        mode_loss = torch.mean(torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma, 1))
        G_loss = -torch.mean(utils.log(D_fake)) + mode_loss
        # Optimize
        G_loss.backward()
        G_solver.step()
        reset_grad()

        """ Plot """
        if (iter+1) == train_loader.dataset.__len__() // batch_size:
            # Print and plot every epoch
            print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}\n'
                  .format(ep, D_loss.data[0], G_loss.data[0], E_loss.data[0]))

            # Reconstruction and generation
            z = utils.to_var(torch.randn(batch_size, z_dim))
            mu, sigma = E(X)
            X_hat = G(z) # randomly generated sample
            X_rec = G(mu) # reconstructed
            eps = utils.to_var(torch.randn(batch_size, z_dim))
            X_rec1 = G(mu + eps * torch.exp(sigma/2.0))
            eps = utils.to_var(torch.randn(batch_size, z_dim))
            X_rec2 = G(mu + eps * torch.exp(sigma/2.0))

            if torch.cuda.is_available():
                print('Mu is {}; Sigma is {}\n'
                      .format(mu.cpu().data.numpy()[0,:], sigma.cpu().data.numpy()[0,:]))
                samples = X_hat.cpu().data.numpy().transpose(0, 2, 3, 1) # 1
                origins = X.cpu().data.numpy().transpose(0, 2, 3, 1) # 2
                recons = X_rec.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
                recons_1 = X_rec1.cpu().data.numpy().transpose(0, 2, 3, 1) # 3
                recons_2 = X_rec2.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
            else:
                print('Mu is {}; Sigma is {}\n'
                      .format(mu.data.numpy()[0,:], sigma.data.numpy()[0,:]))
                samples = X_hat.data.numpy().transpose(0, 2, 3, 1)
                origins = X.data.numpy().transpose(0, 2, 3, 1) # 2
                recons = X_rec.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
                recons_1 = X_rec1.data.numpy().transpose(0, 2, 3, 1) # 3
                recons_2 = X_rec2.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3

            # Save images
            save_images(origins[:4 * 4, :, :, :], [4, 4],
                          '/output/original' + '_epoch%03d' % ep + '.png')
            save_images(samples[:4 * 4, :, :, :], [4, 4],
                          '/output/random' + '_epoch%03d' % ep + '.png')
            save_images(recons[:4 * 4, :, :, :], [4, 4],
                          '/output/reconstructed' + '_epoch%03d' % ep + '.png')
            save_images(recons_1[:4 * 4, :, :, :], [4, 4],
                        '/output/reconstructed_1' + '_epoch%03d' % ep + '.png')
            save_images(recons_2[:4 * 4, :, :, :], [4, 4],
                        '/output/reconstructed_2' + '_epoch%03d' % ep + '.png')

            break
