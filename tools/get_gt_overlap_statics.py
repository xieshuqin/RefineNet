from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import os
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import get_output_dir
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from datasets.json_dataset import JsonDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Return the overlap statistics for ground truth bbox.'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--is_train',
        dest='is_train',
        help='Using training set for calculating statistics',
        action='store_true'
    )
    parser.add_argument(
        '--is_same_cls',
        dest='is_same_cls',
        help='Calculating statics in the same class or not',
        action='store_true'
    )
    # parser.add_argument(
    #     '--imgIds',
    #     dest='imgIds',
    #     help='The imgIds you want to test',
    #     default=None,
    #     type=int
    # )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    if args.is_train:
        print('Evaluating datasets: ', cfg.TRAIN.DATASETS[0])
        ds = JsonDataset(cfg.TRAIN.DATASETS[0])
    else:
        print('Evaluating datasets: ', cfg.TEST.DATASETS[0])
        ds = JsonDataset(cfg.TEST.DATASETS[0])
    # imgIds = [127286]
    roidb = ds.get_roidb(gt=True)
    num_overlaps = ds.get_gt_overlap_statistics(roidb, is_same_cls=args.is_same_cls)
    # ds.get_gt_overlap_for_imgIds(imgIds)

    string_of_class = 'the same class ' if args.is_same_cls else 'all class'
    print('Reporting overlapping statics on ' + string_of_class)
    for k in sorted(num_overlaps):
        print(k, num_overlaps[k])

if __name__ == '__main__':
    main()
