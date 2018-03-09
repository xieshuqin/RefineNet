from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

class RescaleAndDumplicateFeatureOp(object):
    """ Rescale the image feature map and dumplicate them multiple times
    input:  <feature_name> (N x C x H x W)
            <rois>        (M x 5) [batch_id, x1, x2, y1, y2]
    outputs: <feature_name_dump> (M x C x sH x sW), where s = dst_sc / src_sc
    """


    def __init__(self, src_sc, dst_sc):
        self.src_sc = src_sc
        self.dst_sc = dst_sc
    def forward(self, inputs, outputs):
        feat = inputs[0].data
        rois = inputs[1].data

        # rescale features
        feat_resize = []
        num_images = feat.shape[0]
        s = self.dst_sc / self.src_sc

        swap_order = (0, 2, 3, 1)
        feat = feat.transpose(swap_order)
        for i in range(num_images):
            img = feat[i]
            img_resize = cv2.rescale(img, None, None, fx=s, fy=s)
            feat_resize.append(img_resize[np.newaxis, :])

        feat_resize = np.concatenate(feat_resize, axis=0)
        swap_order = (0, 3, 1, 2)
        feat_resize = feat_resize.transpose(swap_order)

        # dumplicate features
        batch_ids = rois[:,0]
        feat_dump = feat_resize[batch_ids,:]

        outputs[0].reshape(feat_dump.shape)
        outputs[0].data[...] = feat_dump


