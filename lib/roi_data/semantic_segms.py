# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Construct minibatches for RefineNet training. Handles the minibatch blobs
that are specific to RefineNet. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from core.config import cfg
import utils.blob as blob_utils
import utils.boxes as box_utils
import utils.segms as segm_utils

logger = logging.getLogger(__name__)

def add_semantic_segms_blobs(blobs, roidb, im_scale, batch_idx, data):
    """ Add Semantic Segmentation Net specidfic blobs to the input blob
        dictionary. Draw all gt polygons to the label
    """
    num_cls = cfg.MODEL.NUM_CLASSES
    rescale_factor = cfg.SEMANTIC_NET.RESCALE_FACTOR
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    polys_gt = [roidb['segms'][i] for i in polys_gt_inds]

    # Define size variables
    inp_h, inp_w = data.shape[2], data.shape[3]
    out_h, out_w = int(inp_h * rescale_factor), int(inp_w * rescale_factor)

    if polys_gt_inds.shape[0] > 0:
        # class label for the mask
        gt_class_labels = roidb['gt_classes'][polys_gt_inds]
        semantic_segms = blob_utils.zeros((num_cls, out_h, out_w), int32=True)
        # narrow scale and size
        scale = im_scale * rescale_factor
        im_h, im_w = roidb['height'], roidb['width']
        im_label_h, im_label_w = int(im_h * scale), int(im_w * scale)

        # add
        for i in range(polys_gt_inds.shape[0]):
            cls_label = gt_class_labels[i]
            poly_gt = polys_gt[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an im_label_h x im_label_w binary image
            mask = segm_utils.polys_to_mask_scaled(poly_gt, im_h, im_w, scale)
            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            semantic_segms[cls_label, 0:im_label_h, 0:im_label_w] = np.maximum(
                semantic_segms[cls_label, 0:im_label_h, 0:im_label_w], mask,
                dtype=np.int32
            )

        semantic_segms = np.reshape(semantic_segms, (1,num_cls*out_h*out_w))

    else:
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).

        # We give it an -1's blob (ignore label)
        semantic_segms = -blob_utils.ones((1, num_cls*out_h*out_w), int32=True)

    blobs['semantic_segms_int32'] = semantic_segms
    blobs['img_rois'] = np.array([batch_idx, 0, 0, inp_w-1, inp_h-1], dtype=np.float32)[np.newaxis, :]
