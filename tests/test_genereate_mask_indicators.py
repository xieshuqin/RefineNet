from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

import utils.c2

from core.config import cfg
from datasets.json_dataset import JsonDataset
import utils.boxes as box_utils
import utils.keypoints as keypoint_utils
import utils.segms as segm_utils

