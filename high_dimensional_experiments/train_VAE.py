from __future__ import (division, print_function, )
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy.random as npr
import numpy as np
from itertools import *

from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

import torch
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

import scipy.misc
import imageio
import matplotlib.gridspec as gridspec
import os, time, pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import Gaussian_Sample_HighD as GS

from scipy.stats import multivariate_normal, entropy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='-1', metavar='GPU',
                    help='set GPU id (default: -1)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='how many epochs to train (default: 100)')
parser.add_argument('--lr-g', type=float, default=1e-4, metavar='LR',
                    help='initial ADAM learning rate of G (default: 5e-4)')
parser.add_argument('--lr-d', type=float, default=1e-4, metavar='LR',
                    help='initial ADAM learning rate of D (default: 5e-4)')
parser.add_argument('--decay', type=float, default=0, metavar='D',
                    help='weight decay or L2 penalty (default: 0)')
parser.add_argument('-z', '--zdim', type=int, default=16, metavar='Z',
                    help='dimension of latent vector (default: 16)')

opt = parser.parse_args()

import os
import sys
import numpy as np
import ite

cuda = 0 if opt.gpu == -1 else 1
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
BS = opt.batch_size
Zdim = opt.zdim
IMAGE_PATH = 'VAE_images'
MODEL_PATH = 'VAE_models'
TEST = 5000

# ===============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from itertools import chain
from torchvision.utils import save_image
from VAE_highd import *

if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists(MODEL_PATH):
    print('mkdir ', MODEL_PATH)
    os.mkdir(MODEL_PATH)

def prog_print(e,b,b_total,loss_g,loss_e):
    sys.stdout.write("\r%3d: [%5d / %5d] G: %.4f E: %.4f" % (e,b,b_total,loss_g,loss_e))
    sys.stdout.flush()

def train():
    # load models
    Gx = GeneratorX()
    Gz = GeneratorZ()

    # load dataset
    # ==========================
    train_data, valid_data, trans_mtx = GS.main()
    train_dataset = GS.Gaussian_Data(train_data)
    valid_dataset = GS.Gaussian_Data(valid_data)

    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=BS,
                            pin_memory= True,
                            shuffle=True)
    validloader = DataLoader(dataset=valid_dataset,
                            batch_size=TEST,
                            pin_memory=True,
                            shuffle=True)

    N = len(dataloader)
    # print(N)

    z = torch.FloatTensor(BS, Zdim).normal_(0, 1)
    z_pred = torch.FloatTensor(TEST, Zdim).normal_(0, 1)
    z_pred = Variable(z_pred)
    noise = torch.FloatTensor(BS, Zdim).normal_(0, 1)

    if cuda:
        Gx.cuda()
        Gz.cuda()
        z, z_pred, noise = z.cuda(), z_pred.cuda(), noise.cuda()


    # optimizer
    optim_g = optim.Adam(chain(Gx.parameters(),Gz.parameters()),
                         lr=opt.lr_g, betas=(.5, .999), weight_decay=opt.decay)

    # train
    # ==========================
    softplus = nn.Softplus()
    mse = nn.MSELoss()
    for epoch in range(opt.epochs):
        Gx.train()
        Gz.train()
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            if cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            z.resize_(batch_size, Zdim).normal_(0, 1)
            noise.resize_(batch_size, Zdim).normal_(0, 1)
            zv = Variable(z)
            noisev = Variable(noise)

            # forward
            encoded = Gz(imgs)
            z_mu, logvar = encoded[:, :Zdim], encoded[:, Zdim:]
            z_enc = z_mu + logvar.mul(0.5).exp() * noisev
            # So encoded[:, Zdim] is log(sigma)
            imgs_fake = Gx(z_enc)

            # compute loss
            beta = 100
            loss_g = beta * torch.mean(torch.sum((imgs_fake - imgs) ** 2, 1))
            loss_e = -0.5 * torch.mean(torch.sum(1 + logvar - z_mu.pow(2) - logvar.exp(), 1))
            loss_ge = loss_g + loss_e

            # backward & update params
            Gx.zero_grad()
            Gz.zero_grad()
            loss_ge.backward()
            optim_g.step()

            prog_print(epoch+1, i+1, N, loss_g.data.item(), loss_e.data.item())

        # generate fake images
        # save_image(Gx(z_pred).data,
        #            os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
        #            nrow=9, padding=1,
        #            normalize=False)
        # save models
        print("-------> Saving models...")
        torch.save(Gx.state_dict(),
                   os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch+1)))
        torch.save(Gz.state_dict(),
                   os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch+1)))

        # evaluate models
        Gx.eval()
        Gz.eval()
        x_eval = Gx(z_pred)
        x_eval = x_eval.data.cpu().numpy()
        for i, (imgs, _) in enumerate(validloader):
            if cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            z_eval = Gz(imgs)
            break
        # print(len(z_eval.data))

        from numpy.random import multivariate_normal, randn
        noise = Variable(torch.FloatTensor(TEST, Zdim).normal_(0, 1).cuda())
        z_sample = z_eval[:, :Zdim] + z_eval[:, Zdim:].mul(0.5).exp() * noise
        z_sample = z_sample.cpu().data.numpy()

        x_mean = np.zeros(256)
        # x_cov = 0.02 ** 2 * np.identity(256) + trans_mtx.T.dot(trans_mtx)
        x_cov = trans_mtx.T.dot(trans_mtx)
        # print(x_cov[:10, :10])

        normal_z_sample = randn(TEST, Zdim)
        conditional_x_sample = z_pred.cpu().data.numpy().dot(trans_mtx)
        normal_x_sample = multivariate_normal(x_mean, x_cov, TEST)

        print(np.max(normal_x_sample))
        # normal_x_sample = randn(TEST, 256)

        # Normality test
        from scipy.stats import normaltest, shapiro
        co = ite.cost.BDKL_KnnKiTi()
        co_easy = ite.cost.BDKL_KnnK()

        print("Our mean is {}, and var is {}".format(np.mean(x_eval[:,0]), np.var(x_eval[:,0])))
        print("True mean is {}, and var is {}".format(np.mean(normal_x_sample[:,0]), np.var(normal_x_sample[:,0])))

        #print("The normal test p-value is: {}".format(normaltest(z_sample.data)))
        print("The shapiro test p-value for z is: {}".format(shapiro(z_sample.data)))
        print("The shapiro test p-value for X is: {}".format(shapiro(x_eval)))

        print("The KL-divergence for z is: {}".format(co.estimation(z_sample, normal_z_sample)))
        print("The KL-divergence for X marginal is: {}".format(co.estimation(x_eval, normal_x_sample)))
        print("The KL-divergence for X conditional is: {}".format(co.estimation(x_eval, conditional_x_sample)))
        # print("The KL-divergence between two X is {}".format(co.estimation(normal_x_sample, conditional_x_sample)))

        #print("The KL-divergence for control is: {}".format(co.estimation(normal_x_sample_2, normal_x_sample)))

        # x_mean = np.dot(z_pred.data.cpu().numpy(), trans_mtx)
        # diff = np.subtract(x_eval, x_mean) ** 2
        #
        # l2 = np.mean(np.sqrt(np.sum(diff, axis=1)))
        # print("The x reconstruction is {}\n".format(l2))


if __name__ == '__main__':
    train()
