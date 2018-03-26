import argparse, os
from zXzGAN import zXzGAN
from gaussian_example import MixedGaussian

"""parsing and configuration"""
def parse_args():
    desc = "zXzGAN pytorch implementation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--root', type=str, default='/output', help='Root of the project')
    parser.add_argument('--dataset', type=str, default='mixed-Gaussian', choices=['mnist', 'fashion-mnist', 'celebA', 'mixed-Gaussian'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrE', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--prior', type=str, default='normal', choices=['normal', 'uniform'])

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

    if args.dataset == 'mixed-Gaussian':
        gan = MixedGaussian(args)
    else:
        gan = zXzGAN(args)

    # launch the graph in a session
    gan.train()


if __name__ == '__main__':
    main()
