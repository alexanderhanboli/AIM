from __future__ import (division, print_function, )
from collections import OrderedDict
from scipy.stats import multivariate_normal

import numpy as np
import numpy.random as npr

from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path
import numpy as np
import itertools
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
import numpy.random as npr
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time
from itertools import *
from torch.utils.data import Dataset
from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import Gaussian_Sample as GS


class Gaussian_Data(Dataset):
    def __init__(self, dataset):
        self.x = torch.from_numpy(np.array(dataset['features'])).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array(dataset['label']))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_data, valid_data = GS.main()
train_dataset = Gaussian_Data(train_data)
valid_dataset = Gaussian_Data(valid_data)

# parameters
batch_size = 64
z_dim = 2
lr = 5e-4
beta1 = 0.5
beta2 = 0.999

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

def to_np(x):
    return x.data.cpu().numpy()

print(train_loader.dataset.__len__())

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
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, z_dim),
        )

    def forward(self, z):
        x = self.fc(z)
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.fc(input)

        return x

class Encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(1024, z_dim)

        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(1024, z_dim),



        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x= self.fc(input)
        mu = self.fc_mu(x)
        sigma = -self.fc_sigma(x)

        return mu,sigma


E = Encoder()
G = Generator()
D = Discriminator()

# E.weight_init(0, 1)
# G.weight_init(0,1)
# D.weight_init(0,1)

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

epoch = 100
for ep in range(1,epoch+1):

    print("Epoch {} started!".format(ep))
    # epoch_start_time = time.time()
    for iter, (X, _) in enumerate(train_loader):

        X = to_var(X)

        """Discriminator"""
        z = to_var(torch.randn(X.size(0), z_dim))

        X_hat = G(z)

        D_real = D(X)
        D_fake = D(X_hat)
        D_loss = -torch.mean(log(D_real) + log(1 - D_fake))
        # Optimize
        D_loss.backward()
        D_solver.step()
        reset_grad()

        """Encoder"""
        z = to_var(torch.randn(X.size(0), z_dim))

        X_hat = G(z)

        z_mu, z_sigma = E(X_hat)
        x_mu, x_sigma = E(X)
        E_loss = torch.mean(torch.mean((z - z_mu) ** 2 * torch.exp(-z_sigma)/2 + z_sigma/2+2, 1))

        # Optimize
        E_loss.backward()
        E_solver.step()
        reset_grad()

        """Generator"""
        #z = to_var(torch.randn(X.size(0), z_dim))

        X_hat = G(z)
        D_fake = D(X_hat)
        z_mu, z_sigma = E(X_hat)
        x_mu, x_sigma = E(X)

        mode_loss = torch.mean(torch.mean((z - z_mu) ** 2 * torch.exp(-z_sigma)/2 + z_sigma/2+2, 1))
        G_loss = -torch.mean(log(D_fake)) + mode_loss
        # Optimize
        G_loss.backward()
        G_solver.step()
        reset_grad()
        if (iter + 1) == train_loader.dataset.__len__() // batch_size:
            print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}\n'
                  .format(ep, D_loss.data[0], G_loss.data[0], E_loss.data[0]))
            G.eval()
            E.eval()
            D.eval()
            Recon = []


            Original = []


            Z = []

            color_vec = []
            for iter, (X, label) in enumerate(valid_loader):

                X = to_var(X)
                label = to_var(label)

                z_mu, z_sigma = E(X)


                X_reconstruc = G(z_mu)


                Original += [x for x in to_np(X)]

                Recon += [x for x in to_np(X_reconstruc)]


                Z += [x for x in to_np(z_mu)]

                color_vec+= [x for x in to_np(label)]
                #print(color_vec)
                #print(Original[:,0])



            #print(Z)
            G.train()
            E.train()
            D.train()
            Original = np.array(Original)
            Recon = np.array(Recon)
            Z = np.array(Z)

            cmap = plt.get_cmap('gnuplot')

            cmap = plt.cm.jet
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(Original[:,0], Original[:,1], c=color_vec, cmap=cmap)


            fig.savefig('./Gaussian_output/Original' + '_epoch%03d' % ep + '.png')
            plt.close()
            #
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(Recon[:,0], Recon[:,1], c = color_vec, cmap=cmap)

            #ax.set_title('X')
            fig.savefig('./Gaussian_output/X_reconstruc' + '_epoch%03d' % ep + '.png')
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(Z[:, 0], Z[:, 1], c=color_vec, cmap=cmap)

            fig.savefig('./Gaussian_output/Z' + '_epoch%03d' % ep + '.png')
            plt.close()
            print("Finished!")
