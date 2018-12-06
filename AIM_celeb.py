import utils, torch, time, os, pickle
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from math import floor
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.pylab import *
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import scipy.misc
import imageio
import matplotlib.gridspec as gridspec
from itertools import *

"""Generator"""
class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist', z_dim = 64, height = None, width = None, pix_level = None):
        super(Generator, self).__init__()
        d = 128
        self.output_dim = z_dim

        self.deconv1 = nn.ConvTranspose2d(z_dim, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        input = input.view(-1, self.output_dim, 1, 1)
        #print(input)
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class FeatureExtractor(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)


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


        return x
"""Encoder"""
class Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist', z_dim = 64, height = None, width = None, pix_level = None):
        super(Encoder, self).__init__()

        self.input_height = height
        self.input_width = width
        self.input_dim = pix_level
        self.output_dim = z_dim
        d = 128
        maxout_pieces = 5
        self.maxout_pieces = maxout_pieces

        # self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        # self.conv3_bn = nn.BatchNorm2d(d * 4)
        #self.conv4 = nn.Conv2d(d * 4, d*8, 4, 2, 1)
        #self.conv4_bn = nn.BatchNorm2d(d*8)
        # self.fc = nn.Sequential(nn.Linear(16384, 1024),
        #                         nn.BatchNorm1d(1024),
        #                         nn.LeakyReLU(0.2))
        # self.fc_mu_max1 =  nn.Sequential(nn.Linear(16384, 200* self.maxout_pieces),
        #                                  nn.BatchNorm1d(200*self.maxout_pieces),
        #                                  nn.LeakyReLU(0.2))
        # self.fc_mu_max2 = nn.Sequential(nn.Linear(200, 200*self.maxout_pieces),
        #                                 nn.BatchNorm1d(200*self.maxout_pieces),
        #                                 nn.LeakyReLU(0.2))
        self.conv4 = nn.Conv2d(d * 8, d * 4, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*4)
        self.conv5 = nn.Conv2d(d * 4, d*4, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*4)
        self.conv6 = nn.Conv2d(d*4, d*2, 2, 1,1)
        self.conv6_bn = nn.BatchNorm2d(d * 2)
        self.conv7 = nn.Conv2d(d*2, d, 1, 1, 0)
        self.conv7_bn = nn.BatchNorm2d(d)


        # self.fc_mu = nn.Sequential(nn.Linear(16384, self.output_dim*10),
        #                            nn.LeakyReLU(0.2),
        #                               nn.Linear(self.output_dim*10, self.output_dim),
        #                               nn.LeakyReLU(0.2),
        #                               nn.Linear(self.output_dim, self.output_dim))

        self.fc_mu = nn.Sequential(nn.Linear(128*2*2, self.output_dim ),
                                   nn.LeakyReLU(0.2),
                                   nn.BatchNorm1d(self.output_dim),
                                   nn.Linear(self.output_dim, self.output_dim))




        # self.fc_sigma_max1 = nn.Sequential(nn.Linear(16384, 200 * self.maxout_pieces),
        #                                 nn.BatchNorm1d(200 * self.maxout_pieces),
        #                                 nn.LeakyReLU(0.2))
        # self.fc_sigma_max2 = nn.Sequential(nn.Linear(200, 200 * self.maxout_pieces),
        #                                 nn.BatchNorm1d(200 * self.maxout_pieces),
        #                                 nn.LeakyReLU(0.2))
        # self.fc_sigma = nn.Sequential(nn.Linear(16384, self.output_dim*10),
        #                            nn.LeakyReLU(0.2),
        #                               nn.Linear(self.output_dim*10, self.output_dim),
        #                               nn.LeakyReLU(0.2),
        #                               nn.Linear(self.output_dim, self.output_dim))
        self.fc_sigma = nn.Sequential(nn.Linear(128*2*2, self.output_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.BatchNorm1d(self.output_dim),
                                   nn.Linear(self.output_dim , self.output_dim))


        # forward method

    def forward(self, x):
       # x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #x = F.leaky_relu(self.conv4(x), 0.2)
        #print(x.view(x.size(0),-1))
       # x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)))
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)))
        #print(x)
        # mu = self.fc_mu_max1(x).view(-1, self.maxout_pieces, 200)
        # mu = torch.max(mu, 1)[0]
        # mu = self.fc_mu_max2(mu).view(-1, self.maxout_pieces, 200)
        # mu = torch.max(mu, 1)[0]
        mu = self.fc_mu(x.view(x.size(0), -1))
        # sigma = self.fc_sigma_max1(x).view(-1, self.maxout_pieces, 200)
        # sigma = torch.max(sigma, 1)[0]
        # sigma = self.fc_sigma_max2(sigma).view(-1, self.maxout_pieces, 200)
        # sigma = torch.max(sigma, 1)[0]
        sigma = self.fc_sigma(x.view(x.size(0), -1))

        return mu, sigma

"""Discriminator"""
class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist', height = None, width = None, pix_level = None):
        super(Discriminator, self).__init__()

        self.input_height = height
        self.input_width = width
        self.input_dim = pix_level
        self.output_dim = 1
        d = 128
        self.maxout_pieces =5


        #self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        #self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)
        # self.fc_max1 = nn.Sequential(nn.Linear(16384, 200 * self.maxout_pieces),
        #                                 nn.BatchNorm1d(200 * self.maxout_pieces),
        #                                 nn.LeakyReLU(0.2))
        # self.fc_max2 = nn.Sequential(nn.Linear(200, 200 * self.maxout_pieces),
        #                                 nn.BatchNorm1d(200 * self.maxout_pieces),
        #                                 nn.LeakyReLU(0.2))
        # self.fc = nn.Linear(200, 1)

        #utils.initialize_weights(self)

    def forward(self, x):
        #x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.fc_max1(x).view(-1, self.maxout_pieces, 200)
        # x = torch.max(x, 1)[0]
        # x = self.fc_max2(x).view(-1, self.maxout_pieces, 200)
        # x = torch.max(x, 1)[0]
        # x = F.sigmoid(self.fc(x))
        x = F.sigmoid(self.conv5(x))
        return x
class zXzGAN_celebA(object):
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
        self.args = args

        # load dataset
        if self.dataset == 'mnist':
            dset = datasets.MNIST('data/mnist', train=True, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
            valid_dset = datasets.MNIST('data/mnist', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'cifar10':
            dset = datasets.CIFAR10(root='data/cifar10', train=True,
                                        download=True,
                                    transform=transforms.Compose([transforms.Scale(64), transforms.ToTensor()]))
            valid_dset = datasets.CIFAR10(root='data/cifar10', train=False, download=True,
                                    transform=transforms.Compose([transforms.Scale(64),transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'fashion-mnist':
            dset = datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor()]))
            valid_dset = datasets.FashionMNIST('data/fashion-mnist', train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor()]))
            self.data_loader = DataLoader(
                dset,
                batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(
                valid_dset,
                batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'celebA':
            # TODO: add test data
            dset = datasets.ImageFolder('./data/resized_celebA/', transform=transforms.ToTensor())
            valid_dats = datasets.ImageFolder('./data/resized_celebA/', transform=transforms.ToTensor())
            num_train = len(dset)
            indices = list(range(num_train))
            valid_size = 0.1
            split = int(np.floor(valid_size * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            self.data_loader = DataLoader(dset, batch_size=self.batch_size,
                                                 sampler=train_sampler)
            self.valid_loader = DataLoader(valid_dats, batch_size=64,
                                       sampler=valid_sampler)

        # image dimensions

        if self.dataset == 'mnist':
            self.height, self.width = dset.train_data.shape[1:3]
            self.pix_level = 1
        elif self.dataset == 'cifar10':
            self.height = 64
            self.width = 64
            self.pix_level = dset.train_data.shape[3]
        elif self.dataset == 'celebA':
            self.height = 64
            self.width = 64
            self.pix_level = 3
        elif len(dset.train_data.shape) == 4:
            self.pix_level = dset.train_data.shape[3]

        if self.dataset == 'celebA':
            self.iter_batch_epoch = floor((0.9 * self.data_loader.dataset.__len__()) // self.batch_size )

        else:
            self.iter_batch_epoch = self.data_loader.dataset.__len__() // self.batch_size

        # networks init
        self.G = Generator(self.dataset, self.z_dim, self.height, self.width, self.pix_level)
        self.E = Encoder(self.dataset, self.z_dim, self.height, self.width, self.pix_level)
        self.D = Discriminator(self.dataset, self.height, self.width, self.pix_level)
        self.FC = FeatureExtractor()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(chain(self.D.parameters(), self.FC.parameters()), lr=args.lrD,
                                      betas=(args.beta1, args.beta2))
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrE, betas=(args.beta1, args.beta2))

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
            self.FC.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.E)
        utils.print_network(self.FC)
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


        print('training start!!')


        start_time = time.time()
        for epoch in range(0, self.epoch):
            # reset training mode of G and E
            self.G_optimizer.param_groups[0]['lr'] = self.args.lrG / np.sqrt(epoch + 1)
            self.D_optimizer.param_groups[0]['lr'] = self.args.lrD / np.sqrt(epoch + 1)
            #self.E_optimizer.param_groups[0]['lr'] = self.args.lrE / np.sqrt(epoch + 1)
            print("learning rate change!")


            epoch_start_time = time.time()
            for iter, (X, _) in enumerate(self.data_loader):
                X = utils.to_var(X)
                if X.size(0) != self.batch_size:
                    break

                """Discriminator"""
                z = utils.to_var(torch.randn(X.size(0), self.z_dim))
                X_hat = self.G(z)



                D_real = self.D(self.FC(X))
                D_fake = self.D(self.FC(X_hat))
                #print(D_fake)
                #print(D_real)
                # D_loss = -torch.mean(utils.log(D_real) + utils.log(1 - D_fake))
                D_loss = self.BCE_loss(D_real, self.y_real_) + self.BCE_loss(D_fake, self.y_fake_)
                self.train_hist['D_loss'].append(D_loss.data[0])
                # Optimize
                D_loss.backward()
                self.D_optimizer.step()
                self.__reset_grad()

                """Encoder"""


                # z = utils.to_var(torch.randn(X.size(0), self.z_dim))
                # X_hat = self.G(z)
                # z_mu, z_sigma = self.E(self.FC(X_hat))
                #
                # E_loss = torch.mean(torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma + 0.919, 1))
                #
                #     # Optimize
                # E_loss.backward()
                # self.E_optimizer.step()
                # self.__reset_grad()
                # self.train_hist['E_loss'].append(E_loss.data[0])

                """Generator"""
                # Use both Discriminator and Encoder to update Generator


                z = utils.to_var(torch.randn(self.batch_size, self.z_dim))
                X_hat = self.G(z)
                D_fake = self.D(self.FC(X_hat))
                z_mu, z_sigma = self.E(self.FC(X_hat))

                mode_loss = torch.mean(
                    torch.mean(0.5 * (z - z_mu) ** 2 * torch.exp(-z_sigma) + 0.5 * z_sigma + 0.919, 1))

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                total_loss = G_loss + mode_loss

                self.train_hist['G_loss'].append(G_loss.data[0])
                self.train_hist['E_loss'].append(mode_loss.data[0])
                # Optimize
                total_loss.backward()
                self.E_optimizer.step()
                self.G_optimizer.step()
                # self.E_optimizer.step()
                self.__reset_grad()

                """ Plot """
                if (iter + 1) == self.iter_batch_epoch:
                    # Print and plot every epoch
                    if (epoch+1) % 5 == 0:
                        self.save(epoch)

                    print('Epoch-{}; D_loss: {:.4}; G_loss: {:.4};  Mode_loss: {:.4}\n'
                          .format(epoch, D_loss.data[0], G_loss.data[0],  mode_loss.data[0]))
                    for iter, (X, _) in enumerate(self.valid_loader):
                        X = utils.to_var(X)
                        self.visualize_results(X, epoch+1)
                        break
                    break

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)


            # Save model every 5 epochs
            if (epoch+1) % 5 == 0:
                self.save(epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 # self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def result_vary(self, epoch):
        image_num = 8
        row = 10
        k = 0
        i = 0
        print(epoch)
        for X, Y in self.valid_loader:
            #print(Y.numpy())

            # if Y.numpy() == k:
            if i == 0:
                images = X
            else:
                images = torch.cat((images, X), 0)
            i += 1
            #                k += 1
            if i == image_num:
                break
        self.load(269)
        X = utils.to_var(images)

        mu, sigma = self.E(self.FC(X))

        for _ in range(0, 1):
            images = X
            for k in range((image_num - 1)):
                eps = utils.to_var(torch.randn(X.size(0), self.z_dim))
                X_rec = self.G(mu + eps * torch.exp(sigma / 2.0))
                images = torch.cat((images, X_rec), 0)

            if torch.cuda.is_available():
                images = images.cpu().data.numpy().transpose(0, 2, 3, 1)  # 1

            else:
                images = images.data.numpy().transpose(0, 2, 3, 1)
            new_images = []

            for i in range(image_num):
                k = i
                for _ in range(image_num):
                    new_images.append(scipy.misc.bytescale(images[k]))
                    k += 8

            images = np.array(new_images)
            save_dir = os.path.join(self.root, self.result_dir, self.dataset, self.model_name)

        print(images.shape)
        if True:
            mb_size = 64
            ss = int(np.sqrt(mb_size))
            fig = plt.figure(figsize=(ss * 2, ss * 2))
            gs = gridspec.GridSpec(ss, ss)
            gs.update(wspace=0.05, hspace=0.05)
            # pdb.set_trace()
            for i, sample in enumerate(images):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample)
                anchors = None
                if anchors:
                    if i in anchors:
                        ax.add_patch(
                            patches.Rectangle(
                                (0.1, 0.1),
                                0.5,
                                0.5,
                                fill=False  # remove background
                            )
                        )

            savefig(os.path.join(save_dir, 'variaonal' + '_epoch%03d' % epoch + '.png'), bbox_inches='tight')

        #utils.generate_animation(save_dir+"/variational", 100)



        self.G.eval()
        self.E.eval()
        self.FC.eval()

    def get_mse(self, epoch):
        #self.load(epoch)
        self.G.eval()
        self.E.eval()
        self.FC.eval()
        critirion = nn.MSELoss()
        count = 0

        for X, _ in self.valid_loader:
            count += 1
            X = utils.to_var(X)
            mu, sigma = self.E(self.FC(X))
            X_hat = self.G(mu)
            loss = (X_hat.view(X_hat.size(0), -1).cpu().data.numpy() - X.view(X.size(0), -1).cpu().data.numpy()) ** 2
            loss = np.mean(loss, 1)
            if count == 1:
                final_loss = loss
            else:
                final_loss = np.concatenate((final_loss, loss), 0)

        print(final_loss.shape)

        print("Final mse mean is %.5f, std is %.5f" % (np.mean(final_loss), np.std(final_loss)))

    def visualize_results(self, X = None, epoch = 0):


        #self.load(epoch)
        print("visualize results...")
        image_num = 64
        batch_size = image_num
        self.get_mse(epoch)
        #row = int(sqrt(image_num))
        row = 8
        nrows = 8
        ncols = 8
        reconstruc = True
        save_dir = os.path.join(self.root, self.result_dir, self.dataset, self.model_name)



        # if X == None:
        #     for X,_ in self.valid_loader:
        #         print(X)
        #         X = utils.to_var(X)
        #         break
        #     self.load(epoch)

        self.G.eval()
        self.E.eval()
        self.FC.eval()


        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        # Reconstruction and generation
        z = utils.to_var(torch.randn(batch_size, self.z_dim))
        mu, sigma = self.E(self.FC(X))


        X_hat = self.G(z) # randomly generated sample
        X_rec = self.G(mu) # reconstructed
        eps = utils.to_var(torch.randn(batch_size, self.z_dim))
        X_rec1 = self.G(mu + eps * torch.exp(sigma/2.0))
        eps = utils.to_var(torch.randn(batch_size, self.z_dim))
        X_rec2 = self.G(mu + eps * torch.exp(sigma/2.0))

        self.G.train()
        self.E.train()
        self.FC.train()




        if torch.cuda.is_available():
            print('Mu is {};\n Sigma is {}\n'
                  .format(mu.cpu().data.numpy()[0,:], sigma.cpu().data.numpy()[0,:]))
            samples = X_hat.cpu().data.numpy().transpose(0, 2, 3, 1) # 1
            origins = X.cpu().data.numpy().transpose(0, 2, 3, 1) # 2
            recons = X_rec.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
            recons_1 = X_rec1.cpu().data.numpy().transpose(0, 2, 3, 1) # 3
            recons_2 = X_rec2.cpu().data.numpy().transpose(0, 2, 3, 1)  # 3
        else:
            print('Mu is {};\n Sigma is {}\n'
                  .format(mu.data.numpy()[0,:], sigma.data.numpy()[0,:]))
            samples = X_hat.data.numpy().transpose(0, 2, 3, 1)
            origins = X.data.numpy().transpose(0, 2, 3, 1) # 2
            recons = X_rec.data.numpy().transpose(0, 2, 3, 1)  # 3
            recons_1 = X_rec1.data.numpy().transpose(0, 2, 3, 1) # 3
            recons_2 = X_rec2.data.numpy().transpose(0, 2, 3, 1)  # 3



        image_recons = []
        for i in range(image_num/2):
            image_recons.append(scipy.misc.bytescale(origins[i, :, :, :]))
            image_recons.append(scipy.misc.bytescale(recons[i, :, :, :]))
        image_recons = np.array(image_recons)


        if reconstruc:


            mb_size = 64
            ss = int(np.sqrt(mb_size))
            fig = plt.figure(figsize=(ss * 2, ss * 2))
            gs = gridspec.GridSpec(ss, ss)
            gs.update(wspace=0.05, hspace=0.05)
            # pdb.set_trace()
            for i, sample in enumerate(image_recons):
                sample = scipy.misc.bytescale(sample)
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample)
                anchors = None
                if anchors:
                    if i in anchors:
                        ax.add_patch(
                            patches.Rectangle(
                                (0.1, 0.1),
                                0.5,
                                0.5,
                                fill=False  # remove background
                            )
                        )


            savefig(os.path.join(save_dir, 'comparison_12' + '_epoch%03d' % epoch + '.png'), bbox_inches='tight')



        # Save images


        # utils.save_images(origins[:image_num, :, :, :], [row,row],
        #                   os.path.join(save_dir, 'original' + '_epoch%03d' % epoch + '.png'))
        # utils.save_images(samples[:image_num, :, :, :], [row,row],
        #                   os.path.join(save_dir, 'random' + '_epoch%03d' % epoch + '.png'))
        # utils.save_images(recons[:image_num, :, :, :], [row,row],
        #                   os.path.join(save_dir, 'reconstructed' + '_epoch%03d' % epoch + '.png'))
        # utils.save_images(recons_1[:image_num, :, :, :], [row,row],
        #                   os.path.join(save_dir, 'reconstructed_1' + '_epoch%03d' % epoch + '.png'))
        # utils.save_images(recons_2[:image_num, :, :, :], [row,row],
        #                   os.path.join(save_dir, 'reconstructed_2' + '_epoch%03d' % epoch + '.png'))




    def save(self, epoch):
        save_dir = os.path.join(self.root, self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_D.pkl'))
        torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_E.pkl'))
        torch.save(self.FC.state_dict(), os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch + '_FC.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            print("Saving the model...")
            pickle.dump(self.train_hist, f)

    def load(self, epoch):
        save_dir = os.path.join(self.root, self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch+ '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch+ '_D.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch+ '_E.pkl')))
        self.FC.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch%03d' % epoch+ '_FC.pkl')))
