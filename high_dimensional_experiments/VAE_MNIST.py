from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
import torch, time, os, pickle
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio
import scipy.misc
import torch.nn.functional as F
import imageio
import matplotlib.gridspec as gridspec
from itertools import *
import Gaussian_Sample_HighD as GS

from scipy.stats import multivariate_normal, entropy
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--log_interval', type=int, default=50, help='manual seed')
parser.add_argument('--hidden_size', type=int, default=16, help='size of z')



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###############   DATASET   ##################
# dset = datasets.MNIST('data/mnist', train=True, download=True,
#                                     transform=transforms.Compose([transforms.ToTensor()]))
# valid_dset = datasets.MNIST('data/mnist', train=False, download=True,
#                                     transform=transforms.Compose([transforms.ToTensor()]))
# data_loader = DataLoader(dset, batch_size=opt.batchSize, shuffle=True)
# valid_loader = DataLoader(valid_dset, batch_size=opt.batchSize, shuffle=True)
###############   MODEL   ##################
class VAE(nn.Module):
    def __init__(self, zd = 16, xd = 256):
        super(VAE,self).__init__()
        # 28 x 28
        n = 64
        self.encod_net = nn.Sequential(
            nn.Linear(xd, zd * 8),

            nn.Dropout(0.2),
            nn.Linear(zd * 8, 4 * zd),


            nn.Dropout(0.2),
            nn.Linear(4 * zd, zd * 4),


            nn.Linear(zd * 4, zd * 2)
        )
        self.gen_net = nn.Sequential(
            nn.Linear(zd, 16 * zd),

            nn.Linear(16 * zd, 16 * zd),

            nn.Linear(16 * zd, 16 * zd),
            nn.Linear(16 * zd, 16 * zd),

            nn.Linear(16 * zd, xd),
        )

        # 28 x 28

    def encoder(self, x):
        # input: noise output: mu and sigma
        # opt.batchSize x 1 x 28 x 28
        output = self.encod_net(x)
        return output[:, :16], output[:, 16:]

    def sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = Variable(eps)
        if(opt.cuda):
            eps = eps.cuda()
        return eps.mul(var).add_(mu)

    def decoder(self, x):
        out = F.tanh(self.gen_net(x))

        return out

    def generate(self, out):
        out = self.decoder(out)
        return out


    def forward(self, x):
        mu, logvar = self.encoder(x)
        out = self.sampler(mu, logvar)
        out = self.decoder(out)
        return out, mu, logvar

model = VAE()
if(opt.cuda):
    model.cuda()
###########   LOSS & OPTIMIZER   ##########
mse = nn.MSELoss()
mse.size_average = False
# if(opt.cuda):
#     bce.cuda()
def LossFunction(out, target, mu, logvar):
    bceloss = mse(out, target)
    #kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kldloss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bceloss + kldloss
optimizer = optim.Adam(model.parameters(),lr=opt.lr)

##########   GLOBAL VARIABLES   ###########
# data = torch.Tensor(opt.batchSize, opt.imageSize * opt.imageSize)
# data = Variable(data)
# if(opt.cuda):
#     data = data.cuda()
###############   TRAINING   ##################
def sample(epoch):
    model.eval()
    eps = torch.FloatTensor(opt.batchSize, opt.hidden_size).normal_()
    eps = Variable(eps)
    if(opt.cuda):
        eps = eps.cuda()
    fake = model.decoder(eps)
    vutils.save_image(fake.data.resize_(opt.batchSize,1,opt.imageSize,opt.imageSize),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)

def sample2d(epoch):
    model.eval()
    eps = torch.FloatTensor(400, opt.hidden_size)
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    for i in range(nx):
        for j in range(ny):
            eps[i*20+j][0] = x_values[i]
            eps[i*20+j][1] = y_values[j]

    eps = Variable(eps)
    if(opt.cuda):
        eps = eps.cuda()
    fake = model.decoder(eps)
    vutils.save_image(fake.data.resize_(400,1,opt.imageSize,opt.imageSize),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True,
                nrow=20)
# def train(epoch):
#     model.train()
#     for i, (images,_) in enumerate(data_loader):
#         model.zero_grad()
#         images = utils.to_var(images)
#         recon, mu, logvar = model(images)
#         loss = LossFunction(recon, images, mu, logvar)
#         loss.backward()
#         optimizer.step()
#         if i % opt.log_interval == 0:
#             sample(epoch)
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, i * len(images), len(data_loader.dataset),
#                 100. * i / len(data_loader),
#                 loss.data[0] / len(images)))
#             get_mse(epoch)


def train():
    # load models

    Zdim = 16
    BS = 32

    model.train()

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
                            batch_size=5000,
                            pin_memory=True,
                            shuffle=True)

    for epoch in range(opt.epochs):
        for i, (images, _) in enumerate(dataloader):


            model.zero_grad()

            images = to_var(images)

            recon, mu, logvar = model(images)

            loss = LossFunction(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

        print("Loss is {}".format(loss.data[0]))

        z_pred = torch.FloatTensor(5000, Zdim).normal_(0, 1)
        z_pred = to_var(z_pred)

        x_eval = model.generate(z_pred)
        x_eval = x_eval.data.cpu().numpy()
        for i, (images, _) in enumerate(validloader):


            imgs = to_var(images)
            recon, z_eval, logvar = model(imgs)
            break
        # print(len(z_eval.data))



        pk = multivariate_normal.pdf(z_eval.data[:, :Zdim], mean=np.zeros(16))
        print("The z entropy is {}".format(entropy(pk)))

        x_mean = np.dot(z_pred.data.cpu().numpy(), trans_mtx)
        diff = np.subtract(x_eval, x_mean) ** 2

        l2 = np.mean(np.sqrt(np.sum(diff, axis=1)))
        print("The x reconstruction is {}\n".format(l2))



def get_mse(epoch):
    model.eval()

    count = 0
    for X,_ in valid_loader:
        count += 1
        X = to_var(X)
        X_hat, mu, logvar = model(X)
        loss = (X_hat.view(X_hat.size(0), -1).cpu().data.numpy() - X.view(X.size(0), -1).cpu().data.numpy())**2
        loss = np.mean(loss, 1)
        if count == 1:
            final_loss = loss
        else:
            final_loss = np.concatenate((final_loss, loss), 0)

    print(final_loss.shape)


    print( "Final mse mean is %.5f, std is %.5f" %(np.mean(final_loss), np.std(final_loss)))

# for epoch in range(1, opt.niter + 1):
#     train(epoch)
# if(opt.hidden_size == 2):
#     sample2d(epoch)
train()
