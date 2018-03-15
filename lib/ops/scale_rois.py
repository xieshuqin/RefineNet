from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import utils.boxes as box_utils
from caffe2.proto import caffe2_pb2

class ScaleRoIsSingleOp(object):
    """ Scale the rois by a factor up_scale and
        then clip them within the image boundary

        inputs: [data, rois]
        outputs: [scaled_rois]
    """

    def __init__(self, up_scale):
        self.up_scale = up_scale

    def forward(self, inputs, outputs):
        data = inputs[0].data
        rois = inputs[1].data
        up_scale = self.up_scale
        height, width = data.shape[2], data.shape[3]

        bboxes = rois[:, 1:5]
        batch_ids = rois[:, [0]]
        # up-scale the bboxes and clip to image boundary
        # pad bboxes and narrow to the feature map scale
        pad_bboxes = box_utils.expand_boxes_by_scale(bboxes, up_scale)
        pad_bboxes = box_utils.clip_boxes_to_image(pad_bboxes, height, width)

        # add the batch_ids to the rois
        pad_rois = np.hstack((batch_ids, pad_bboxes)).astype(np.int32)

        outputs[0].reshape(pad_rois.shape)
        outputs[0].data[...] = pad_rois


class ScaleRoIsFPNOp(object):
    """ Scale the rois by a factor up_scale and
        then clip them within the image boundary

        inputs: [data, rois_fpn<min>, ... , rois_fpn<max>, rois_idx_restore_int32 ]
        outputs: [pad_rois_fpn<min>, ... , pad_rois_fpn<max>, pad_rois_idx_restore_int32 ]
    """
    def __init__(self, k_min, k_max, up_scale):
        self.k_min = k_min
        self.k_max = k_max
        self.up_scale = up_scale

    def forward(self, inputs, outputs):
        data = inputs[0].data
        k_min = self.k_min
        k_max = self.k_max
        up_scale = self.up_scale
        height, width = data.shape[2], data.shape[3]

        for lvl in range(k_min, k_max + 1):
            rois = inputs[1 + lvl - k_min].data # skip the 'data' blob
            bboxes = rois[:, 1:5]
            batch_ids = rois[:, [0]]
            # up-scale the bboxes and narrow down to image boundary
            pad_bboxes = box_utils.expand_boxes_by_scale(bboxes, up_scale)
            pad_bboxes = box_utils.clip_boxes_to_image(pad_bboxes, height, width)
            # add the batch_ids to the rois
            pad_rois = np.hstack((batch_ids, pad_bboxes)).astype(np.int32)

            outputs[lvl - k_min].reshape(pad_rois.shape)
            outputs[lvl - k_min].data[...] = pad_rois

        # copy rois_idx_restore_int32 to the scale_rois_idx_restore_int32
        # A little surgery for int32 type requirement
        rois_idx_restore_int32 = inputs[-1].data
        outputs[-1].init(list(rois_idx_restore_int32.shape), caffe2_pb2.TensorProto.INT32)
        outputs[-1].reshape(rois_idx_restore_int32.shape)
        outputs[-1].data[...] = rois_idx_restore_int32.astype(np.int32)
