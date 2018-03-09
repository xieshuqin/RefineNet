from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from core.config import cfg
from datasets import json_dataset
import modeling.FPN as fpn
import roi_data.fast_rcnn
import utils.blob as blob_utils

class GenerateIndicators(object):
    def __init(self):

    def forward(self, inputs, outputs):
        fpn = inputs[0].data
        rois = inputs[1].data
