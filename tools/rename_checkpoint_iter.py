import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Rename checkpoint iteration. Given the original number of GPUs and iterations, as well as the new number of GPUs you will use to train your model, this function convert the checkpoint to the corresponding new iteration'
    )
    parser.add_argument(
            '--DIR',
            dest='DIR',
            help='The dir where you want to convert the checkpoint',
            default=None,
            type=str
    )
    parser.add_argument(
            '--orig_num_iters',
            dest='orig_num_iters',
            help='The original number of iterations',
            default=None,
            type=float
    )
    parser.add_argument(
            '--orig_num_gpus',
            dest='orig_num_gpus',
            help='The original number of GPUs used for training',
            default=None,
            type=int
    )
    parser.add_argument(
            '--new_num_gpus',
            dest='new_num_gpus',
            help='New number of GPUs you will used to train your model.',
            default=None,
            type=int
    )
    if len(sys.argv) in [1,2,3,4]:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    args = parse_args()
    rename_checkpoint(
        folder=args.DIR,
        orig_num_gpus=args.orig_num_gpus,
        orig_num_iters=args.orig_num_iters,
        new_num_gpus=args.new_num_gpus
    )

def rename_checkpoint(folder, orig_num_gpus, orig_num_iters, new_num_gpus):
    folder = "\\\\".join(folder.split('\\'))
    assert os.path.isdir(folder), folder + " is not a dir "

    total_iters = orig_num_gpus * orig_num_iters
    new_num_iters = int(total_iters / new_num_gpus)
    for f in os.listdir(folder):
        if f.startswith('model_iter') and f.endswith('.pkl'):
            cur_iters = float(f[f.find('model_iter')+10:f.find('.pkl')]) + 1
            new_iters = int(cur_iters * orig_num_gpus / new_num_gpus)-1
            new_name = 'model_iter' + str(new_iters) + '.pkl'
            os.rename(os.path.join(folder, f), os.path.join(folder, new_name))

if __name__ == '__main__':
    main()

