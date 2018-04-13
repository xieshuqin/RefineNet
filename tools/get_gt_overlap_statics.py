from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import get_output_dir
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from datasets.json_datasets import JsonDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Return the overlap statistics for ground truth bbox.'
    )
    arser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    print('Evaluating datasets: ', cfg.TRAIN.DATASETS)
    ds = JsonDataset(cfg.TRAIN.DATASETS)
    roidb = ds.get_roidb(gt=True)
    num_overlaps = ds.get_gt_overlap_statistics(roidb, is_same_cls=True)

    print('num_overlaps: ', num_overlaps)

if __name__ == '__main__':
    main()
