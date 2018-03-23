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

NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BETA1 = 0.8
BATCH_SIZE = 128
MONITORING_BATCH_SIZE = 500
PRIORS = None

MEANS = [np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                           range(-4, 5, 2))]
VARIANCES = [0.05 ** 2 * np.eye(len(mean)) for mean in MEANS]


class GaussianMixture():
    """ Toy dataset containing points sampled from a gaussian mixture distribution.
    The dataset contains 3 sources:
    * features
    * label
    * densities
    """

    def __init__(self, num_examples, means=None, variances=None, priors=None,
                 **kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', config.default_seed)
            rng = np.random.RandomState(seed)

        gaussian_mixture = GaussianMixtureDistribution(means=means,
                                                       variances=variances,
                                                       priors=priors,
                                                       rng=rng)
        self.means = gaussian_mixture.means
        self.variances = gaussian_mixture.variances
        self.priors = gaussian_mixture.priors

        features, labels = gaussian_mixture.sample(nsamples=num_examples)
        densities = gaussian_mixture.pdf(x=features)

        self.data = {
            'features': features,
            'label': labels,
            'density': densities
        }

    def get_data(self):
        return self.data

def create_gaussian_mixture_data(batch_size, monitoring_batch_size,
                                         means=None, variances=None, priors=None,
                                         rng=None, num_examples=100000,
                                         sources=('features', )):
    train_set = GaussianMixture(num_examples=num_examples, means=means,
                                variances=variances, priors=priors,
                                rng=rng, sources=sources)

    valid_set = GaussianMixture(num_examples=num_examples,
                                means=means, variances=variances,
                                priors=priors, rng=rng, sources=sources)

    return train_set, valid_set
def as_array(obj):
    """Converts to ndarray of specified dtype"""
    return np.asarray(obj)

class GaussianMixtureDistribution(object):
    """ Gaussian Mixture Distribution
    Parameters
    ----------
    means : tuple of ndarray.
       Specifies the means for the gaussian components.
    variances : tuple of ndarray.
       Specifies the variances for the gaussian components.
    priors : tuple of ndarray
       Specifies the prior distribution of the components.
    """

    def __init__(self, means=None, variances=None, priors=None, rng=None, seed=None):

        if means is None:
            means = map(lambda x:  10.0 * as_array(x), [[0, 0],
                                                        [1, 1],
                                                        [-1, -1],
                                                        [1, -1],
                                                        [-1, 1]])
        # Number of components
        self.ncomponents = len(means)
        self.dim = means[0].shape[0]
        self.means = means
        # If prior is not specified let prior be flat.
        if priors is None:
            priors = [1.0/self.ncomponents for _ in range(self.ncomponents)]
        self.priors = priors
        # If variances are not specified let variances be identity
        if variances is None:
            variances = [np.eye(self.dim) for _ in range(self.ncomponents)]
        self.variances = variances

        assert len(means) == len(variances), "Mean variances mismatch"
        assert len(variances) == len(priors), "prior mismatch"

        if rng is None:
            rng = npr.RandomState(seed=seed)
        self.rng = rng

    def _sample_prior(self, nsamples):
        return self.rng.choice(a=self.ncomponents,
                               size=(nsamples, ),
                               replace=True,
                               p=self.priors)

    def sample(self, nsamples):
        # Sampling priors
        samples = []
        fathers = self._sample_prior(nsamples=nsamples).tolist()
        for father in fathers:
            samples.append(self._sample_gaussian(self.means[father],
                                                 self.variances[father]))
        return as_array(samples), as_array(fathers)

    def _sample_gaussian(self, mean, variance):
        # sampling unit gaussians
        epsilons = self.rng.normal(size=(self.dim, ))

        return mean + np.linalg.cholesky(variance).dot(epsilons)

    def _gaussian_pdf(self, x, mean, variance):
        return multivariate_normal.pdf(x, mean=mean, cov=variance)

    def pdf(self, x):
        "Evaluates the the probability density function at the given point x"
        pdfs = map(lambda m, v, p: p * self._gaussian_pdf(x, m, v),
                   self.means, self.variances, self.priors)
        return reduce(lambda x, y: x + y, pdfs, 0.0)



data = create_gaussian_mixture_data(
        batch_size=BATCH_SIZE, monitoring_batch_size=MONITORING_BATCH_SIZE,
        means=MEANS, variances=VARIANCES, priors=PRIORS)
train_data = data[0].get_data()
valid_datat = data[1].get_data()

class Gaussian_Data(Dataset):
    def __init__(self, dataset):
        self.x = torch.from_numpy(np.array(dataset['features']))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

train_dataset = Gaussian_Data(train_data)
valid_dataset = Gaussian_Data(valid_data)

# parameters
batch_size = 128
z_dim = 256
X_height = 218 # 28
X_width = 178 # 28
h_dim = 64
lr = 5e-5
beta1 = 0.5
beta2 = 0.999

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
print(train_loader.dataset.__len__())

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

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


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.fc = nn.Sequential(
#             nn.Linear(z_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 256 * (X_height // 4) * (X_width // 4)),
#             nn.BatchNorm1d(256 * (X_height // 4) * (X_width // 4)),
#             nn.ReLU(),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 3, 4, 2, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, z):
#         x = self.fc(z)
#         x = x.view(-1, 256,  (X_height // 4),  (X_width // 4))
#         x = self.deconv(x)
#         return x

class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

class Encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

        self.fc_mu = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = x.view(x.size(0),-1)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu,sigma



# class Encoder(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
#     def __init__(self, dataset = 'mnist'):
#         super(Encoder, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 128, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 256, 4, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#         )
#         self.fc_mu = nn.Sequential(
#             nn.Linear(256 *  (X_height // 4) *  (X_width // 4), 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, z_dim),
#         )
#         self.fc_sigma = nn.Sequential(
#             nn.Linear(256 *  (X_height // 4) *  (X_width // 4), 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, z_dim),
#         )
#
#     def forward(self, input):
#         x = self.conv(input)
#         x = x.view(-1, 256 * (X_height // 4) * (X_width // 4))
#         mu = self.fc_mu(x)
#         sigma = self.fc_sigma(x)
#         return mu, sigma

# class Discriminator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
#     def __init__(self, dataset = 'mnist'):
#         super(Discriminator, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 128, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 256, 4, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(256 *  (X_height // 4) *  (X_width // 4), 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, z_dim),
#             nn.Linear(z_dim, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input):
#         x = self.conv(input)
#         x = x.view(-1, 256 * (X_height // 4) * (X_width // 4))
#         x = self.fc(x)
#         return x

E = Encoder()
G = Generator()
D = Discriminator()

# cuda
if torch.cuda.is_available():
    print("CUDA")
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
        # Sample z and data
        X = to_var(X)






        """Discriminator"""
        z = to_var(torch.randn(batch_size, z_dim))
        z_ = z.view(-1, z_dim, 1, 1)
        X_hat = G(z_)

        D_real = D(X)
        D_fake = D(X_hat)
        D_loss = -torch.mean(log(D_real) + log(1 - D_fake))
        # Optimize
        D_loss.backward()
        D_solver.step()
        reset_grad()

        """Encoder"""
        z = to_var(torch.randn(batch_size, z_dim))
        z_ = z.view(-1, z_dim, 1, 1)
        X_hat = G(z_)
        z_mu, z_sigma = E(X_hat)
        E_loss = torch.mean(torch.mean((z - z_mu) ** 2 * torch.exp(-z_sigma)/2 + z_sigma/2+0.7, 1))  # - loglikehood
        # Optimize
        E_loss.backward()
        E_solver.step()
        reset_grad()

        """Generator"""
        z = to_var(torch.randn(batch_size, z_dim))
        z_ = z.view(-1, z_dim, 1, 1)
        X_hat = G(z_)
        D_fake = D(X_hat)
        z_mu, z_sigma = E(X_hat)
        mode_loss = torch.mean(torch.mean((z - z_mu) ** 2 * torch.exp(-z_sigma)/2 + z_sigma/2+0.7, 1))
        G_loss = -torch.mean(log(D_fake)) + mode_loss
        # Optimize
        G_loss.backward()
        G_solver.step()
        reset_grad()

        """ Plot """
        if (iter + 1) == train_loader.dataset.__len__() // batch_size:
            print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}\n'
                  .format(ep, D_loss.data[0], G_loss.data[0], E_loss.data[0]))
            mu, sigma = E(X)
            z_ = z.view(-1, z_dim, 1, 1)
            X_hat = G(z_)
            mu_= mu.view(-1, z_dim, 1, 1)
            X_rec = G(mu_)
            eps = to_var(torch.randn(batch_size, z_dim))
            z1 = mu + eps * torch.exp(sigma / 2.0)
            z1 = z1.view(-1, z_dim, 1, 1)
            X_rec1 = G(z1)
            eps = to_var(torch.randn(batch_size, z_dim))
            z2 = mu + eps * torch.exp(sigma / 2.0)
            z2 = z2.view(-1, z_dim, 1, 1)
            X_rec2 = G(z2)

            if torch.cuda.is_available():
                samples = X_hat.cpu().data.numpy().transpose(0, 2, 3, 1) # 1
                origins = X.cpu().data.numpy().transpose(0, 2, 3, 1) # 2
                recons = X_rec.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
                recons_1 = X_rec1.cpu().data.numpy().transpose(0, 2, 3, 1) # 3
                recons_2 = X_rec2.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
            else:
                samples = X_hat.data.numpy().transpose(0, 2, 3, 1)
                origins = X.data.numpy().transpose(0, 2, 3, 1) # 2
                recons = X_rec.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
                recons_1 = X_rec1.data.numpy().transpose(0, 2, 3, 1) # 3
                recons_2 = X_rec2.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3

            # Save images
            save_images(origins[:4 * 4, :, :, :], [4, 4],
                          './celeb_output/original' + '_epoch%03d' % ep + '.png')
            save_images(samples[:4 * 4, :, :, :], [4, 4],
                          './celeb_output/random' + '_epoch%03d' % ep + '.png')
            save_images(recons[:4 * 4, :, :, :], [4, 4],
                          './celeb_output/reconstructed' + '_epoch%03d' % ep + '.png')
            save_images(recons_1[:4 * 4, :, :, :], [4, 4],
                        './celeb_output/reconstructed_1' + '_epoch%03d' % ep + '.png')
            save_images(recons_2[:4 * 4, :, :, :], [4, 4],
                        './celeb_output/reconstructed_2' + '_epoch%03d' % ep + '.png')

            break
