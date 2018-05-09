import argparse, os
from LAI import LAI
from LAI_cl import LAI_cl
from dcLAI import dcLAI
from LAI_mix_gaussian_cl import LAI_mg_cl
from LAI_mix_gaussian_fGAN import LAI_mg_cl_fgan
from LAI_mix_gaussian import LAI_mg

"""parsing and configuration"""
def parse_args():
    desc = "LAI pytorch implementation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--root', type=str, default='/output', help='Root of the project')
    parser.add_argument('--model_name', type=str, default='LAI', help='Model name')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist', 'fashion-mnist', 'svhn', 'cifar10', 'celebA', 'mixed-Gaussian'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    parser.add_argument('--lrG', type=float, default=1e-4)
    parser.add_argument('--lrD', type=float, default=1e-4)
    parser.add_argument('--lrE', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=bool, default=False, choices=[True, False])
    parser.add_argument('--grad_clip', type=bool, default=False, choices=[True, False])
    parser.add_argument('--grad_clip_val', type=float, default=3.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--prior', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--load_model', type=bool, default=False, choices=[True, False])

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
        gan = LAI_mg_cl_fgan(args)
    elif args.model_name == 'dcLAI':
        gan = dcLAI(args)
    else:
        gan = LAI_cl(args)

    # launch the graph in a session
    gan.train()


if __name__ == '__main__':
    main()
