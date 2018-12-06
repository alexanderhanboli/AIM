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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='-1', metavar='GPU',
                    help='set GPU id (default: -1)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='how many epochs to train (default: 100)')
parser.add_argument('--lr-g', type=float, default=2e-4, metavar='LR',
                    help='initial ADAM learning rate of G (default: 2e-4)')
parser.add_argument('--lr-d', type=float, default=1e-5, metavar='LR',
                    help='initial ADAM learning rate of D (default: 1e-5)')
parser.add_argument('--decay', type=float, default=0, metavar='D',
                    help='weight decay or L2 penalty (default: 0)')
parser.add_argument('-z', '--zdim', type=int, default=128, metavar='Z',
                    help='dimension of latent vector (default: 128)')

opt = parser.parse_args()

import os
import sys
import numpy as np
cuda = 0 if opt.gpu == -1 else 1
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
BS = opt.batch_size
Zdim = opt.zdim
IMAGE_PATH = 'AIM_images'
MODEL_PATH = 'AIM_models'

# ===============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from itertools import chain
from torchvision.utils import save_image
from AIM_highd import *

if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists(MODEL_PATH):
    print('mkdir ', MODEL_PATH)
    os.mkdir(MODEL_PATH)

def prog_ali(e,b,b_total,loss_g,loss_d,dx,dgz):
    sys.stdout.write("\r%3d: [%5d / %5d] G: %.4f D: %.4f D(x,Gz(x)): %.4f D(Gx(z),z): %.4f" % (e,b,b_total,loss_g,loss_d,dx,dgz))
    sys.stdout.flush()

def train():
    # load models
    Gx = GeneratorX(zd=Zdim)
    Gz = GeneratorZ(zd=Zdim)
    Dx = DiscriminatorX(zd=Zdim)

    # load dataset
    # ==========================
    train_data, valid_data = GS.main()
    train_dataset = GS.Gaussian_Data(train_data)
    valid_dataset = GS.Gaussian_Data(valid_data)

    data_loader = DataLoader(dataset=train_dataset,
                            batch_size=self.batch_size,
                            pin_memory= True,
                            shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=self.batch_size,
                            pin_memory=True,
                            shuffle=False)

    N = len(dataloader)

    z = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)
    z_pred = torch.FloatTensor(81, Zdim, 1, 1).normal_(0, 1) # 9 by 9
    z_pred = Variable(z_pred)
    noise = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)

    if cuda:
        Gx.cuda()
        Gz.cuda()
        Dx.cuda()
        z, z_pred, noise = z.cuda(), z_pred.cuda(), noise.cuda()


    # optimizer
    optim_g = optim.Adam(chain(Gx.parameters(),Gz.parameters()),
                         lr=opt.lr_g, betas=(.5, .999), weight_decay=opt.decay)
    optim_d = optim.Adam(chain(Dx.parameters()),
                         lr=opt.lr_d, betas=(.5, .999), weight_decay=opt.decay)

    # train
    # ==========================
    softplus = nn.Softplus()
    for epoch in range(opt.epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            if cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            z.resize_(batch_size, Zdim, 1, 1).normal_(0, 1)
            noise.resize_(batch_size, Zdim, 1, 1).normal_(0, 1)
            zv = Variable(z)
            noisev = Variable(noise)

            # forward
            imgs_fake = Gx(zv)
            encoded = Gz(imgs)
            # reparametrization trick
            z_enc = encoded[:, :Zdim] + encoded[:, Zdim:].exp() * noisev # So encoded[:, Zdim] is log(sigma)
            z_mu, z_sigma = encoded[:, :Zdim], encoded[:, Zdim:]
            d_true = Dx(imgs)
            d_fake = Dx(imgs_fake)
            # reconstruction


            # compute loss
            loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))
            loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))
            loss_e = torch.mean(torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma + 0.5 * np.log(2*np.pi), 1))
            loss_ge = loss_g + loss_e

            # backward & update params
            Dx.zero_grad()
            loss_d.backward(retain_graph=True)
            optim_d.step()
            Gx.zero_grad()
            Gz.zero_grad()
            loss_ge.backward()
            optim_g.step()

            prog_ali(epoch+1, i+1, N, loss_g.data[0], loss_d.data[0], d_true.data.mean(), d_fake.data.mean())

        # generate fake images
        save_image(Gx(z_pred).data,
                   os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
                   nrow=9, padding=1,
                   normalize=False)
        # save models
        torch.save(Gx.state_dict(),
                   os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch+1)))
        torch.save(Gz.state_dict(),
                   os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch+1)))
        torch.save(Dx.state_dict(),
                   os.path.join(MODEL_PATH, 'Dx-%d.pth'  % (epoch+1)))
        print()


if __name__ == '__main__':
    train()
