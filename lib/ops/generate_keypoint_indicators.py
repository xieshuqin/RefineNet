from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.keypoints as keypoint_utils


class GenerateKeypointIndicatorsOp(object):
    """ Generate keypoint indicators.
        A function similar to GenerateLocalMaskIndicator but deal with special
        issues for keypoint heatmap.

        Since the input keypoint heatmap is usually a one-hot heatmap,
        resizing it to a smaller resolution will result in a darken heatmap or
        even an all-zero heatmap.
        To avoid this problem, instead of resizing the original heatmap,
        we get the position with maximum value from the original heatmap and
        convert it to the new rois. Then we draw a one-hot heatmap based on
        the converted position and the new rois.

        inputs: [data, keypoint_probs, keypoint_rois]
        outputs: [keypoint_indicators]
    """

    def __init__(self, up_scale, resolution):
        self.up_scale = up_scale
        self.resolution = resolution

    def forward(self, inputs, outputs):
        data = inputs[0].data
        keypoint_probs = inputs[1].data
        keypoint_rois = inputs[2].data

        # output indicator resolution
        M = self.resolution
        up_scale = self.up_scale
        num_rois = keypoint_rois.shape[0]
        num_keypoints = keypoint_probs.shape[1]

        # first expand the keypoint rois
        height, width = data.shape[2], data.shape[3]
        pad_rois = box_utils.expand_boxes_by_scale(keypoint_rois[:, 1:5], up_scale)
        pad_rois = box_utils.clip_boxes_to_image(pad_rois, height, width)

        # get keypoint predictions and their probs
        # output shape is (#rois, 3, #keypoints) and 3 means (x, y, prob)
        pred_rois = keypoint_utils.probs_to_keypoints(keypoint_probs, keypoint_rois)
        # map keypoint position to the pad_rois
        # output shape is (#rois, #keypoints), locations flatter out
        locations_on_pad_rois, _ = keypoint_utils.keypoints_to_heatmap_labels(
            pred_rois, pad_rois, M
        )
        locations_on_pad_rois = locations_on_pad_rois.astype(np.int32)

        # and now generate keypoint indicators
        keypoint_indicators = blob_utils.zeros((num_rois, num_keypoints, M**2))
        for i in range(num_rois):
            locations = locations_on_pad_rois[i] # shape (#keypoints, )
            for k in range(num_keypoints):
                keypoint_indicators[i, k, locations[k]] = pred_rois[i, 2, k]

        # and reshape to 4 dimension
        keypoint_indicators = keypoint_indicators.reshape(
            (num_rois, num_keypoints, M, M)
        )

        outputs[0].reshape(keypoint_indicators.shape)
        outputs[0].data[...] = keypoint_indicators

    def backward(self, inputs, outputs):
        # We don't back-propagate for this layer. So just pass a zero array.
        data = inputs[0]
        grad_data = outputs[0]

        grad_data.reshape(data.shape)
        grad_data.data[...] = np.zeros(data.shape, dtype=np.float32)






