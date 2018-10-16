from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from core.config import cfg
import utils.boxes as box_utils

# Test for time
import time

class GenerateGlobalMaskIndicatorsOp(object):
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

        # some data processing
        swap_order = (0, 2, 3, 1)
        mask_probs = mask_probs.transpose(swap_order)
        mask_rois = (mask_rois[:, 1:5]*scale).astype(np.int32) # convert rois to int32

        for i in range(num_rois):
            mask_local = mask_probs[i]
            roi = mask_rois[i]
            shape = (roi[2]-roi[0]+1, roi[3]-roi[1]+1) # (roi_w, roi_h)
            mask_local_resize = cv2.resize(mask_local, shape)
            if mask_local.shape[2] == 1:
                mask_local_resize = mask_local_resize[:,:,np.newaxis]

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


class GenerateLocalMaskIndicatorsOp(object):
    """ Generate mask indicators in a local area.
        We enlarge the roi by a up_scale factor, converting the
        local mask_probs to this enlarged box, then resized the
        box to a fixed resolution, serving as the indicator

        inputs: [data, mask_probs, mask_rois]
        outputs: [mask_indicators]
    """

    def __init__(self, up_scale, resolution):
        self.up_scale = up_scale
        self.resolution = resolution

    def forward(self, inputs, outputs):
        tic = time.time()

        data = inputs[0].data
        mask_probs = inputs[1].data
        mask_rois = inputs[2].data

        # whether using binary threshold for indicator
        if cfg.REFINENET.USE_PERCENTTHRES_INDICATOR:
            mask_probs_reshape = mask_probs.reshape(mask_probs.shape[0],
                                                    mask_probs.shape[1],-1)
            thres = int(cfg.REFINENET.PERCENTINDICATOR_THRES * \
                        mask_probs_reshape.shape[2])
            mask_probs_sort = np.argsort(-mask_probs_reshape, axis=2)[:,:,:thres]
            mask_binary = np.zeros(mask_probs_reshape.shape, dtype=np.float32)
            for i in range(mask_probs_sort.shape[0]):
                for j in range(mask_probs_sort.shape[1]):
                    mask_binary[i,j,mask_probs_sort[i,j]] = 1.
            mask_binary = mask_binary.reshape(mask_probs.shape)
            mask_probs *= mask_binary
        elif cfg.REFINENET.USE_THRES_INDICATOR:
            mask_binary = np.array(
                mask_probs > cfg.REFINENET.INDICATOR_THRES, dtype=np.float32
            )
            mask_probs *= mask_binary
        # output indicator resolution
        M = self.resolution
        up_scale = self.up_scale
        num_cls = mask_probs.shape[1]
        num_rois = mask_rois.shape[0]
        mask_indicators = np.zeros((num_rois, M, M, num_cls), dtype=np.float32)

        # preparing data
        height, width = data.shape[2], data.shape[3]
        mask_probs_NHWC = mask_probs.transpose((0,2,3,1))
        rois = mask_rois[:, 1:5] # ignore batch_id
        pad_rois = box_utils.expand_boxes(rois, up_scale)
        pad_rois = box_utils.clip_boxes_to_image(pad_rois, height, width)

        # calculate converted coordinates
        converted_coords = box_utils.convert_coordinate(rois, pad_rois, M)
        for i in range(num_rois):
            mask_prob = mask_probs_NHWC[i]
            coords = converted_coords[i]
            shape = (coords[2]-coords[0]+1, coords[3]-coords[1]+1) # w,h
            if shape[0] < 1 or shape[1] < 1:
                continue
            mask_prob_resize = cv2.resize(mask_prob, shape)
            if mask_prob_resize.shape[2] == 1:
                mask_prob_resize = mask_prob_resize[:, :, np.newaxis]
            mask_indicators[i, coords[1]:coords[3]+1, coords[0]:coords[2]+1] = \
                mask_prob_resize

        swap_order = (0, 3, 1, 2)
        mask_indicators = mask_indicators.transpose(swap_order)

        outputs[0].reshape(mask_indicators.shape)
        outputs[0].data[...] = mask_indicators

        toc = time.time()
        print('Running time for GenerateLocalMaskIndicatorsOp is {:.5f}'.format(toc - tic))

    def backward(self, inputs, outputs):
        # We don't back-propagate for this layer. So just pass a zero array.
        data = inputs[0]
        grad_data = outputs[0]

        grad_data.reshape(data.shape)
        grad_data.data[...] = np.zeros(data.shape, dtype=np.float32)

        #mask_probs = inputs[1]
        #grad_mask_probs = outputs[1]
        #grad_mask_probs.reshape(mask_probs.shape)
        #grad_mask_probs.data[...] = np.zeros(mask_probs.shape, dtype=np.float32)








