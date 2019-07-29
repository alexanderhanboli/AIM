import utils, torch, time, os, pickle
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio
import scipy.misc
import imageio
import matplotlib.gridspec as gridspec
from itertools import *

"""Generator"""


class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset='mnist', z_dim=64, height=None, width=None, pix_level=None):
        super(Generator, self).__init__()

        self.input_height = height
        self.input_width = width
        self.input_dim = z_dim
        self.output_dim = pix_level

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
        # print(x)
        return x


"""Discriminator"""


class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist', height=None, width=None, pix_level=None):
        super(Discriminator, self).__init__()

        self.input_height = height
        self.input_width = width
        self.output_dim = 1
        # self.conv = nn.Sequential(
        #     nn.Conv2d(64, 128, 4, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1),
        # )

        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 64),
            nn.Linear(64, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, x):
        # x = self.conv(x)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        return x


"""FeatureExtrator"""


class Feature(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist', height=None, width=None, pix_level=None):
        super(Feature, self).__init__()

        self.input_height = height
        self.input_width = width
        self.input_dim = pix_level

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        return x


####
# 5: increase the capicity of encoder
# 6: increase the dimension of z(64 -> 128)
####

class LAI_MNIST_BASE(object):
    def __init__(self, args):
        # parameters
        self.root = args.root
        self.epoch = args.epoch
        self.sample_num = 16
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.z_dim = args.z_dim
        self.model_name = args.model_name
        self.load_model = args.load_model
        self.args = args

        # load dataset
        if self.dataset == 'mnist':
            dset = datasets.MNIST('data/mnist', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))
            valid_dset = datasets.MNIST('data/mnist', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dset, batch_size=50, shuffle=True)
        elif self.dataset == 'emnist':
            dset = datasets.EMNIST('data/emnist', split='balanced', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
            valid_dset = datasets.EMNIST('data/emnist', split='balanced', train=False, download=True,
                                         transform=transforms.Compose([transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'cifar10':
            dset = datasets.CIFAR10(root='data/mnist', train=True,
                                    download=True, transform=transforms.Compose([transforms.ToTensor()]))
            valid_dset = datasets.CIFAR10(root='data/mnist', train=False, download=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'svhn':
            dset = datasets.SVHN(root='data/svhn', split='train',
                                 download=True, transform=transforms.Compose([transforms.ToTensor()]))
            valid_dset = datasets.SVHN(root='data/svhn', split='test', download=True,
                                       transform=transforms.Compose([transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'fashion-mnist':
            dset = datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor()]))
            valid_dset = datasets.FashionMNIST('data/fashion-mnist', train=False, download=True,
                                               transform=transforms.Compose(
                                                   [transforms.ToTensor()]))
            self.data_loader = DataLoader(
                dset,
                batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(
                valid_dset,
                batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'celebA':
            # TODO: add test data
            dset = utils.load_celebA('data/celebA', transform=transforms.Compose(
                [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size,
                                          shuffle=True)

        # image dimensions
        if self.dataset == 'svhn':
            self.height, self.width = dset.data.shape[2:4]
            self.pix_level = dset.data.shape[1]
        else:
            self.height, self.width = dset.train_data.shape[1:3]
            if len(dset.train_data.shape) == 3:
                self.pix_level = 1
            # elif self.dataset == 'cifar10':
            #     self.height = 2* self.height
            #     self.width = 2 * self.width
            #     self.pix_level = dset.train_data.shape[3]
            elif len(dset.train_data.shape) == 4:
                self.pix_level = dset.train_data.shape[3]

        print("Data shape is height:{}, width:{}, and pixel level:{}\n".format(self.height, self.width, self.pix_level))

        # networks init
        self.G = Generator(self.dataset, self.z_dim, self.height, self.width, self.pix_level)
        #self.E = Encoder(self.dataset, self.z_dim, self.height, self.width, self.pix_level)
        self.D = Discriminator(self.dataset, self.height, self.width, self.pix_level)
        self.FC = Feature(self.dataset, self.height, self.width, self.pix_level)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1 * 1.2, args.beta2))
        self.D_optimizer = optim.Adam(chain(self.D.parameters(), self.FC.parameters()), lr=args.lrD,
                                      betas=(args.beta1 * 1.2, args.beta2))
        #self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrE, betas=(args.beta1 * 1.2, args.beta2))

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            #self.E.cuda()
            self.FC.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.FC)
        print('-----------------------------------------------')

        # load in saved model
        # self.checkpoint = 0
        # self.load_model = True
        # if self.load_model:
        #     self.checkpoint = 299
        #     print("Loading model..."+str(self.checkpoint))
        #     self.load(self.checkpoint)

    def __reset_grad(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()
        #self.E_optimizer.zero_grad()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['E_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if torch.cuda.is_available():
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(
                torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        # self.D.train()
        print('training start!!')
        start_time = time.time()
        self.load(149)
        for epoch in range(150, self.epoch):
            self.G_optimizer.param_groups[0]['lr'] = self.args.lrG / np.sqrt(epoch + 1)
            self.D_optimizer.param_groups[0]['lr'] = self.args.lrD / np.sqrt(epoch + 1)
            # reset training mode of G and E

            epoch_start_time = time.time()

            D_err = []
            G_err = []
            # learning rate decay
            # if (epoch+1) % 20 == 0:
            #     self.G_optimizer.param_groups[0]['lr'] /= 2
            #     self.D_optimizer.param_groups[0]['lr'] /= 2
            #     self.E_optimizer.param_groups[0]['lr'] /= 2
            #     print("learning rate change!")
            # self.G_optimizer.param_groups[0]['lr'] /= np.sqrt(epoch+1)
            # self.D_optimizer.param_groups[0]['lr'] /= np.sqrt(epoch+1)
            # self.E_optimizer.param_groups[0]['lr'] /= np.sqrt(epoch+1)
            # print("learning rate change!")


            for iter, (X, _) in enumerate(self.data_loader):

                X = utils.to_var(X)

                """Discriminator"""
                z = utils.to_var(torch.randn(self.batch_size, self.z_dim))
                X_hat = self.G(z)
                D_real = self.D(self.FC(X))
                D_fake = self.D(self.FC(X_hat))
                D_loss = self.BCE_loss(D_real, self.y_real_) + self.BCE_loss(D_fake, self.y_fake_)
                self.train_hist['D_loss'].append(D_loss.data[0])
                D_err.append(D_loss.data[0])
                # Optimize
                D_loss.backward()
                self.D_optimizer.step()
                self.__reset_grad()

                """Generator"""
                # Use both Discriminator and Encoder to update Generator
                z = utils.to_var(torch.randn(self.batch_size, self.z_dim))
                X_hat = self.G(z)
                D_fake = self.D(self.FC(X_hat))

                # E_loss = torch.mean(
                #     torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) +
                #                0.5 * z_sigma + 0.919, 1))

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                total_loss = G_loss
                self.train_hist['G_loss'].append(G_loss.data[0])
                G_err.append(G_loss.data[0])
                #E_err.append(E_loss.data[0])
                # Optimize
                total_loss.backward()
                self.G_optimizer.step()
                #self.E_optimizer.step()
                self.__reset_grad()

                """ Plot """
                if (iter + 1) == self.data_loader.dataset.__len__() // self.batch_size:
                    # Print and plot every epoch
                    print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4}\n'
                          .format(epoch, np.mean(D_err), np.mean(G_err)))
                    for iter, (X, _) in enumerate(self.valid_loader):
                        X = utils.to_var(X)
                        self.visualize_results(X, epoch + 1)
                        break

                    break

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            # Save model
            if (epoch + 1) % 5 == 0:
                self.save(epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        # self.save(epoch)

        # Generate animation of reconstructed plot
        utils.generate_animation(
            self.root + '/' + self.result_dir + '/' + self.dataset + '/' + self.model_name + '/reconstructed',
            self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.root, self.save_dir, self.dataset, self.model_name),
                        self.model_name)

    def visualize_results(self, X=None, epoch=0):
        print("visualize results...")
        image_num = 100
        batch_size = image_num / 2
        # row = int(sqrt(image_num))
        row = 10
        nrows = 8
        ncols = 8
        reconstruc = True
        save_dir = os.path.join(self.root, self.result_dir, self.dataset, self.model_name)


        self.G.eval()
        #self.E.eval()
        self.FC.eval()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Reconstruction and generation
        z = utils.to_var(torch.randn(batch_size * 2, self.z_dim))


        X_hat = self.G(z)  # randomly generated sample


        self.G.train()
        #self.E.train()
        self.FC.train()

        if torch.cuda.is_available():

            samples = X_hat.cpu().data.numpy().transpose(0, 2, 3, 1)  # 1
            origins = X.cpu().data.numpy().transpose(0, 2, 3, 1)  # 2

        else:

            samples = X_hat.data.numpy().transpose(0, 2, 3, 1)
            origins = X.data.numpy().transpose(0, 2, 3, 1)  # 2





        # Save images


        utils.save_images(origins[:image_num, :, :, :], [row, row],
                          os.path.join(save_dir, 'original' + '_epoch%03d' % epoch + '.png'))
        utils.save_images(samples[:image_num, :, :, :], [row, row],
                          os.path.join(save_dir, 'random' + '_epoch%03d' % epoch + '.png'))




    def generate_images(self, epoch):
        checkpoint = 399
        self.load(epoch)


        self.G.eval()

        save_dir = os.path.join(self.root, self.result_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        batch_size = 100
        total_num = 50000
        images = []

        for k in range(1):
            for iter in range(total_num / batch_size):
                # Reconstruction and generation
                z = utils.to_var(torch.randn(batch_size, self.z_dim))
                X_hat = self.G(z)  # randomly generated sample
                if torch.cuda.is_available():
                    samples = X_hat.cpu().data
                else:
                    samples = X_hat.data

                samples = samples.view(samples.size(0), -1)

                if iter == 0:
                    images = samples
                else:
                    images = torch.cat((images, samples), 0)

            print(images.size())
            sio.savemat(save_dir + '/' + '{}.mat'.format(str(epoch + 1).zfill(3)), {'images': images.numpy()})
        self.G.train()
        #print(sio.loadmat(save_dir + '/' + '{}.mat'.format(str(1).zfill(3))))

    def save(self, epoch):
        save_dir = os.path.join(self.root, self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_D.pkl'))
        #torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_E.pkl'))
        torch.save(self.FC.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_FC.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            print("Saving the model...")
            pickle.dump(self.train_hist, f)

    def load(self, epoch=99):
        save_dir = os.path.join(self.root, self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_D.pkl')))
        #self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_E.pkl')))
        self.FC.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_FC.pkl')))