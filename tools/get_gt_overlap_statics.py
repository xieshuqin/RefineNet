from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from dataset.json_datasets import JsonDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Return the overlap statistics for ground truth bbox.'
    )
    parser.add_argument(
            '--DATASET',
            dest='DATASET',
            help='The dataset you want to evaluate',
            default=None,
            type=str
    )
    if len(sys.argv) in [1,2,3,4]:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = JsonDataset(args.DATASET)
    roidb = ds.get_roidb(gt=True)
    num_overlaps = ds.get_gt_overlap_statistics(roidb, is_same_cls=True)

    print(num_overlaps)

if __name__ == '__main__':
    main()
