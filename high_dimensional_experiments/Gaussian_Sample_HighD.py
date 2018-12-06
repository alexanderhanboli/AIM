from __future__ import (division, print_function, )
from collections import OrderedDict
from scipy.stats import multivariate_normal

import numpy as np
import numpy.random as npr

from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path

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

import scipy.misc
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time
from itertools import *
from torch.utils.data import Dataset
from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from functools import reduce
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

class Gaussian_Data(Dataset):
    def __init__(self, dataset):
        self.x = torch.from_numpy(np.array(dataset['features'])).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array(dataset['label']))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def main():
    data = create_gaussian_mixture_data(
        batch_size=BATCH_SIZE, monitoring_batch_size=MONITORING_BATCH_SIZE,
        means=MEANS, variances=VARIANCES, priors=PRIORS)

    train_data = data[0].get_data()
    valid_data = data[1].get_data()
    return train_data, valid_data
