from __future__ import (division, print_function, )
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy.random as npr
import numpy as np
import utils
import itertools

from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset

import scipy.misc
import imageio
import matplotlib.gridspec as gridspec
import os, time, pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import Gaussian_Sample as GS


"""Generator"""
class Generator(nn.Module):
    def __init__(self, G_dim = 400):
        super(Generator, self).__init__()

        self.input_dim = 2
        self.hid_dim = G_dim
        self.output_dim = 2

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.output_dim)
        )
        utils.initialize_weights(self)

    def forward(self, z):
        x = self.fc(z)
        return x

"""Encoder"""
class Encoder(nn.Module):
    def __init__(self, E_dim = 400):
        super(Encoder, self).__init__()

        self.input_dim = 2
        self.hid_dim = E_dim
        self.output_dim = 2

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.output_dim),
        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.output_dim),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x= self.fc(input)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu,sigma

"""Discriminator"""
class Discriminator(nn.Module):
    def __init__(self, D_dim = 200, maxout_pieces = 5):
        super(Discriminator, self).__init__()

        self.input_dim = 2
        self.hid_dim = D_dim
        self.maxout_pieces = maxout_pieces
        self.output_dim = 1

        # TODO
        # This is a naive maxout implementation. Should be updated later.
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim * self.maxout_pieces),
        )
        self.fcmax1 = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim * self.maxout_pieces),
        )
        self.fcmax2 = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim * self.maxout_pieces),
        )
        self.fo = nn.Sequential(
            nn.Linear(self.hid_dim, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, self.maxout_pieces, self.hid_dim)
        x = torch.max(x, 1)[0] # maxout layer 1
        x = self.fcmax1(x)
        x = x.view(-1, self.maxout_pieces, self.hid_dim)
        x = torch.max(x, 1)[0] # maxout layer 2
        x = self.fcmax2(x)
        x = x.view(-1, self.maxout_pieces, self.hid_dim)
        x = torch.max(x, 1)[0] # maxout layer 3
        x = self.fo(x)

        return x

class MixedGaussian(object):
    def __init__(self, args):
        # parameters
        self.root = args.root
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = "MixedGaussianExample"
        self.z_dim = 2
        self.prior = args.prior

        """Generate data"""
        train_data, valid_data = GS.main()
        train_dataset = GS.Gaussian_Data(train_data)
        valid_dataset = GS.Gaussian_Data(valid_data)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        self.G = Generator()
        self.E = Encoder()
        self.D = Discriminator()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrE, betas=(args.beta1, args.beta2))

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.E)
        print('-----------------------------------------------')

    def __reset_grad(self):
        self.E_optimizer.zero_grad()
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['E_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if torch.cuda.is_available():
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            # reset training mode of G and E
            self.G.train()
            self.E.train()
            epoch_start_time = time.time()
            for iter, (X, _) in enumerate(self.train_loader):
                X = utils.to_var(X)

                """Discriminator"""
                z = utils.generate_z(self.batch_size, self.z_dim, self.prior)
                X_hat = self.G(z)
                D_real = self.D(X)
                D_fake = self.D(X_hat)
                D_loss = self.BCE_loss(D_real, self.y_real_) + self.BCE_loss(D_fake, self.y_fake_)
                self.train_hist['D_loss'].append(D_loss.data[0])
                # Optimize
                D_loss.backward()
                self.D_optimizer.step()
                self.__reset_grad()

                """Encoder"""
                z = utils.generate_z(self.batch_size, self.z_dim, self.prior)
                X_hat = self.G(z)
                z_mu, z_sigma = self.E(X_hat)
                # - loglikehood
                E_loss = torch.mean(torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma + 0.9189 * self.z_dim, 1))
                self.train_hist['E_loss'].append(E_loss.data[0])
                # Optimize
                E_loss.backward()
                self.E_optimizer.step()
                self.__reset_grad()

                """Generator"""
                # Use both Discriminator and Encoder to update Generator
                z = utils.generate_z(self.batch_size, self.z_dim, self.prior)
                X_hat = self.G(z)
                D_fake = self.D(X_hat)
                z_mu, z_sigma = self.E(X_hat)
                mode_loss = torch.mean(torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma + 0.9189 * self.z_dim, 1))
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                total_loss = G_loss + mode_loss
                self.train_hist['G_loss'].append(G_loss.data[0])
                # Optimize
                total_loss.backward()
                self.G_optimizer.step()
                self.__reset_grad()

                """Plot"""
                if (iter + 1) == self.train_loader.dataset.__len__() // self.batch_size:
                    print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}\n'
                          .format(epoch, D_loss.data[0], G_loss.data[0], E_loss.data[0]))

                    self.visualize_results(epoch+1)

                    break

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            # Save model every 5 epochs
            if epoch % 5 == 0:
                self.save()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def visualize_results(self, epoch):
        self.G.eval()
        self.E.eval()

        save_dir = os.path.join(self.root, self.result_dir, 'mixed_gaussian', self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Store results
        Recon = []
        Original = []
        Z = []
        Random = []
        color_vec = []

        for iter, (X, label) in enumerate(self.valid_loader):
            z = utils.to_var(torch.randn(self.batch_size, self.z_dim))
            X = utils.to_var(X)
            label = utils.to_var(label)
            z_mu, z_sigma = self.E(X)
            X_reconstruc = self.G(z_mu)
            X_random = self.G(z)

            Original += [x for x in utils.to_np(X)]
            Recon += [x for x in utils.to_np(X_reconstruc)]
            Z += [x for x in utils.to_np(z_mu)]
            Random += [x for x in utils.to_np(X_random)]
            color_vec+= [x for x in utils.to_np(label)]

        Original = np.array(Original)
        Recon = np.array(Recon)
        Z = np.array(Z)
        Random = np.array(Random)

        cmap = plt.get_cmap('gnuplot')
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(Original[:,0], Original[:,1], c=color_vec, cmap=cmap)
        fig.savefig(os.path.join(save_dir, 'X_original' + '_epoch%03d' % epoch + '.png'))
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(Recon[:,0], Recon[:,1], c=color_vec, cmap=cmap)
        fig.savefig(os.path.join(save_dir, 'X_reconstruc' + '_epoch%03d' % epoch + '.png'))
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(Random[:,0], Random[:,1])
        fig.savefig(os.path.join(save_dir, 'X_random' + '_epoch%03d' % epoch + '.png'))
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(Z[:,0], Z[:,1], c=color_vec, cmap=cmap)
        fig.savefig(os.path.join(save_dir, 'Z_mu' + '_epoch%03d' % epoch + '.png'))
        plt.close()

    def save(self):
        save_dir = os.path.join(self.root, self.save_dir, 'mixed_gaussian', self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_E.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            print("Saving the model...")
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.root, self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
