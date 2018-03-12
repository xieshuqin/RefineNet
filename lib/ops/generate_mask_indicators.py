from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

class GenerateMaskIndicatorsOp(object):
    """ See detector.py for more detailed documents.
    Input blobs: [data, mask_probs, mask_rois]
    Output blobs: [mask_indicators]

    """
    def __init__(self, scale=1/16.):
        self.scale = scale

    def forward(self, inputs, outputs):
        data = inputs[0].data
        mask_probs = inputs[1].data
        mask_rois = inputs[2].data
        scale = self.scale # spatial_scale

        height, width = int(data.shape[2]*scale), int(data.shape[3]*scale)
        num_rois = mask_rois.shape[0]
        num_cls = mask_probs.shape[1]
        mask_indicators = np.zeros(shape=(num_rois, height, width, num_cls))
        print('mask_indicators shape', height, width)

        # some data processing
        swap_order = (0, 2, 3, 1)
        mask_probs = mask_probs.transpose(swap_order)
        mask_rois = (mask_rois[:, 1:5]*scale).astype(np.int32) # convert rois to int32

        for i in range(num_rois):
            mask_local = mask_probs[i]
            roi = mask_rois[i]
            shape = (roi[2]-roi[0]+1, roi[3]-roi[1]+1) # (roi_w, roi_h)
            mask_local_resize = cv2.resize(mask_local, shape)
            mask_indicators[i,roi[1]:roi[3]+1,roi[0]:roi[2]+1,:] = mask_local_resize

        swap_order = (0, 3, 1, 2)
        mask_indicators = mask_indicators.transpose(swap_order)

        outputs[0].reshape(mask_indicators.shape)
        outputs[0].data[...] = mask_indicators

    def backward(self, inputs, outputs):
        # We don't back-propagate for this layer. So just pass a zero array.
        data = inputs[0]
        grad_in = outputs[0]

        grad_in.reshape(data.shape)
        grad_in.data[...] = np.zeros(data.shape, dtype=np.float32)

