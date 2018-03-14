from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import utils.boxes as box_utils

class PoolingIndicatorFeatureSingleOp(object):
    """ 
    Generate indicator feature by 
    1. Up scale the rois 
    2. Pool the feature map from the pad_rois
    3. Resize to MxM, where M is defined in cfg.REFINENET.RESOLUTION

    input:  <feature_name> (N x C x H x W)
            <rois>        (M x 5) [batch_id, x1, x2, y1, y2]
    outputs: <feature_name_dump> (M x C x sH x sW), where s = dst_sc / src_sc
    """


    def __init__(self, spatial_scale, up_scale, resolution):
        self.spatial_scale = spatial_scale
        self.up_scale = up_scale
        self.resolution = resolution

    def forward(self, inputs, outputs):
        feat = inputs[0].data
        rois = inputs[1].data
        up_scale = self.up_scale
        spatial_scale = self.spatial_scale
        M = self.resolution
        num_rois = rois.shape[0]

        # convert from NCHW to NHWC
        feat = feat.transpose((0, 2, 3, 1))
        feat_h, feat_w = feat.shape[1], feat.shape[2]

        # pad rois and narrow to the feature map scale
        pad_rois = box_utils.expand_boxes_by_scale(rois[:,1:5], up_scale)
        pad_rois = (pad_rois * spatial_scale).astype(np.int32)
        pad_rois = box_utils.clip_boxes_to_image(pad_rois, feat_h, feat_w)

        # abstact feature from the pad_rois
        pad_roi_feats = np.zeros((num_rois, M, M, feat.shape[3]))
        batch_idx = rois[:,0]
        for i in range(num_rois):
            batch_id = batch_idx[i]
            pad_roi = pad_rois[i]
            pad_roi_feat = feat[batch_id, pad_roi[1]:pad_roi[3]+1, pad_roi[0]:pad_roi[2]+1, :]
            pad_roi_feat_resize = cv2.resize(pad_roi_feat, (M, M))
            pad_roi_feats[i] = pad_roi_feat_resize

        pad_roi_feats.transpose((0, 3, 1, 2))

        outputs[0].reshape(pad_roi_feats.shape)
        outputs[0].data[...] = pad_roi_feats


    def backward(self, inputs, outputs):
        """ Currently, we didn't back-propagate into the feature. 
        Thus, we pass a zero-array as the gradient 
        """
        feature = inputs[0].data
        grad_feature = outputs[0]

        grad_feature.reshape(feature.shape)
        grad_feature.data[...] = np.zeros(feature.shape, dtype=np.float32)


class PoolingIndicatorFeatureFPNOp(object):
    """ Rescale the FPN feature map and dumplicate them multiple times
    input:  [fpn_<min>, <rois>_fpn<min>,
                        .... ,
             fpn_<max>, <rois>_fpn<max> ]

            fpn shape:  (N x C x H x W)
            rois shape: (M x 5) [batch_id, x1, x2, y1, y2]

    outputs:  (M x C x sH x sW), where s = dst_sc / src_sc
    """


    def __init__(self, k_min, k_max, spatial_scale, up_scale, resolution):
        self.k_min = k_min
        self.k_max = k_max
        self.spatial_scales = spatial_scale
        self.up_scale = up_scale
        self.resolution = resolution
        self.num_fpn_lvls = self.k_max - self.k_min + 1

    def forward(self, inputs, outputs):

        up_scale = self.up_scale
        M = self.resolution
        for lvl in range(self.num_fpn_lvls):
            feat = inputs[2*lvl].data
            rois = inputs[2*lvl+1].data

            num_rois = rois.shape[0]
            spatial_scale = self.spatial_scales[lvl]

            # convert from NCHW to NHWC
            feat = feat.transpose((0, 2, 3, 1))
            feat_h, feat_w = feat.shape[1], feat.shape[2]

            # pad rois and narrow to the feature map scale
            pad_rois = box_utils.expand_boxes_by_scale(rois[:,1:5], up_scale)
            pad_rois = (pad_rois * spatial_scale).astype(np.int32)
            pad_rois = box_utils.clip_boxes_to_image(pad_rois, feat_h, feat_w)

            # abstact feature from the pad_rois
            pad_roi_feats = np.zeros((num_rois, M, M, feat.shape[3]))
            batch_idx = rois[:,0]
            for i in range(num_rois):
                batch_id = batch_idx[i]
                pad_roi = pad_rois[i]
                pad_roi_feat = feat[batch_id, pad_roi[1]:pad_roi[3]+1, pad_roi[0]:pad_roi[2]+1, :]
                pad_roi_feat_resize = cv2.resize(pad_roi_feat, (M, M))
                pad_roi_feats[i] = pad_roi_feat_resize

            pad_roi_feats.transpose((0, 3, 1, 2))

            outputs[lvl].reshape(pad_roi_feats.shape)
            outputs[lvl].data[...] = pad_roi_feats


    def backward(self, inputs, outputs):
        """ Currently, we don't back-propagate into the feature map
        Thus we pass a zero array as the gradient
        """
        for lvl in range(self.num_fpn_lvls):
            fpn_feat = inputs[2*lvl].data
            grad_fpn_feat = outputs[lvl]

            grad_fpn_feat.reshape(fpn_feat.shape)
            grad_fpn_feat.data[...] = np.zeros(fpn_feat.shape, dtype=np.float32)



