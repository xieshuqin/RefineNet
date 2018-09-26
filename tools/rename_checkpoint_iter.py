import argparse
import sys
import os
import re

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
        '--old_n_gpus',
        dest='old_n_gpus',
        help='Old number of GPUs used for training',
        default=None,
        type=int
    )
    parser.add_argument(
        '--new_n_gpus',
        dest='new_n_gpus',
        help='New number of GPUs used for training',
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
        old_n_gpus=args.old_n_gpus,
        new_n_gpus=args.new_n_gpus
    )

def rename_checkpoint(folder, old_n_gpus, new_n_gpus):
    folder = "\\\\".join(folder.split('\\'))
    assert os.path.isdir(folder), folder + " is not a dir "

    # Find and rename all checkpoints
    files = os.listdir(folder)
    for f in files:
        iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
        if len(iter_string) > 0:
            old_iter = int(iter_string[0]) + 1
            new_iter = int(old_iter * old_n_gpus / new_n_gpus) - 1
            new_name = 'model_iter' + str(new_iter) + '.pkl'
            os.rename(os.path.join(folder, f), os.path.join(folder, new_name))

if __name__ == '__main__':
    main()

