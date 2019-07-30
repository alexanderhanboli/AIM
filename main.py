import argparse, os
#from AIM import AIM
#from GAIM import GAIM
#from dcAIM import dcAIM
from gaussian_example import MixedGaussian
from gan_mix_gaussian import GAN_MixedGaussian
from AIM_celeb import zXzGAN_celebA
from AIM_cifar_10 import zXzGAN
from AIM_f_cifar_10 import f_zXzGAN
from AIM_MNIST import AIM_MNIST
# from AIM_imagenet import zXzGAN_IMAGE
# from cifar10_Baseline import zXzGAN_baseline
# from AIM_SLEEP import zXzGAN_sleep
from AIM_MNIST_BASE import AIM_MNIST_BASE
# from MNIST_Z_Class import AIM_MNIST_CLASS
# from AIM_imagenet_base import zXzGAN_IMAGE_base
# from AIM_MNIST_CYCLE import AIM_MNIST_CYCLE
# from AIM_cifar10_cycle import zXzGAN_cycle
# from ALI_Gaussian import ALI_mg
# from VEEGAN import VEEGAN_mg
from AIM_f_MNIST import AIM_f_MNIST
#from cifar_10_32 import zXzGAN
#from AIM_cifar_10 import zXzGAN
#from AIM_mix_gaussian_cl import AIM_mg_cl
#from AIM_mix_gaussian import AIM_mg

import torch
import numpy as np

"""parsing and configuration"""
def parse_args():
    desc = "AIM pytorch implementation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--root', type=str, default='./output', help='Root of the project')
    parser.add_argument('--model_name', type=str, default='AIM_f_MNIST', help='Model name')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist',
                                                                         'svhn', 'cifar10', 'celebA', 'image-net', 'mixed-Gaussian'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=300, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    parser.add_argument('--lrG', type=float, default=2e-5)
    parser.add_argument('--lrD', type=float, default=2e-5)
    parser.add_argument('--lrE', type=float, default=2e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--prior', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--generate_images', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--sleep', action='store_true', default=False)
    parser.add_argument('--icp_eval', action='store_true', default=False)
    parser.add_argument('--mainfold', action='store_true', default=False)
    parser.add_argument('--varies', action='store_true', default=False)
    parser.add_argument('--uniform_sampling', action='store_true', default=False)

    parser.add_argument('--seed_random', default=42, type=int)


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # set seed
    torch.manual_seed(args.seed_random)
    np.random.seed(args.seed_random)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed_random)

    if args.dataset == 'mixed-Gaussian':
        if args.model_name == 'AIM':
            gan = MixedGaussian(args)
        elif args.model_name == 'GAN':
            gan = GAN_MixedGaussian(args)
         #gan = ALI_mg(args)
         # gan = VEEGAN_mg(args)
    elif args.dataset == 'cifar10':
        if args.model_name == 'zXzGAN_baseline':
            gan = zXzGAN_baseline(args)
            print("CIFAR BASELINE")
        elif args.model_name == 'zXzGAN_cycle':
            print("cycle")
            gan = zXzGAN_cycle(args)
        elif args.model_name == 'zXzGAN_f':
            print("cifar f-AIM")
            gan = f_zXzGAN(args)
        else:
            gan = zXzGAN(args)
    elif args.dataset == 'image-net':
        if args.model_name == 'zXzGAN_baseline':
            print("Imagenet Baseline")
            gan = zXzGAN_IMAGE_base(args)
        else:
            gan = zXzGAN_IMAGE(args)
    elif args.dataset == 'celebA':
        if args.sleep:
            gan = zXzGAN_sleep(args)
        else:
            gan = zXzGAN_celebA(args)
    elif args.dataset == 'mnist':
        if args.model_name == 'AIM_MNIST_base':
            gan = AIM_MNIST_BASE(args)
        elif args.model_name == 'AIM_MNIST_cycle':
            print("cycle")
            gan = AIM_MNIST_CYCLE(args)
        elif args.model_name == 'AIM_MNIST_CLASS':
            print("icml rebutal")
            gan = AIM_MNIST_CLASS(args)
        elif args.model_name == 'AIM_f_MNIST':
            print("f-GAN AIM\n")
            gan = AIM_f_MNIST(args)
        else:
            gan = AIM_MNIST(args)

    if args.generate_images:
        print("Generate Images")
        gan.generate_images(49)
    elif args.visualize:
        #for epoch in range(109,30,10):
        epoch = 599
        print(epoch)
        gan.visualize_results(epoch)
        #gan.get_mse(179)
    elif args.varies:
        for epoch in range(0,10,1):
        #epoch = 199
            gan.result_vary(epoch)
    elif args.icp_eval == True:
        for epoch in range(5, 55, 5):
            print(epoch)
            #epoch = 3
            gan.get_mse(epoch-1)
            #gan.generate_images(epoch)
    elif args.mainfold:
        gan.manifold(399)
    elif args.uniform_sampling:
        gan.uniform()

    else:
        print("train")
        gan.train()
        # for epoch in range(200, 300, 20):
        #     print(epoch)
        #     #epoch = 3
        #     gan.get_mse(epoch-1)








if __name__ == '__main__':
    main()
