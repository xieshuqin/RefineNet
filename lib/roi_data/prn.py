""" Construct minibatches for PRN training. Handles the minibatch blobs 
that are specific to IoUNet. Other blobs that are generic to Fast/er R-CNN, 
etc are handled by their respective roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from core.config import cfg

logger = logging.getLogger(__name__)

def get_prn_blob_names(is_training=True):
    blob_names = []
    if is_training:
        blob_names += ['prn_labels_int32', 'roi_needs_refine_int32']

    return blob_names

def add_prn_blobs(blobs_out, blobs_in):
    """ Add PRN specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    num_cls = cfg.MODEL.NUM_CLASSES
    iou_thres = cfg.PRN.IOU_THRESHOLD

    fg_inds = np.where(blobs_in['labels_int32'] > 0)[0]
    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        fg_labels = blobs_in['labels_int32'][fg_inds]
        # if below threshold, then set labels to 1, otherwise 0
        prn_labels = (blobs_in['mask_ious'] < iou_thres).astype(np.int32)
        # and set roi_needs_refine same as prn_labels
        roi_needs_refine = prn_labels.copy()

    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs_in['labels_int32'] == 0)[0]
        # We give it an -1's blob (ignore label)
        prn_labels = -blob_utils.ones((1, ), int32=True)
        # We label it with class = 0 (background)
        fg_labels = blob_utils.zeros((1, ))
        # and set roi_needs_refine to be 1
        roi_needs_refine = blob_utils.ones((1, ), int32=True)

    if cfg.PRN.CLS_SPECIFIC_LABEL:
        prn_labels = _expand_to_class_specific_prn_targets(prn_labels, fg_labels)

    blobs_out['prn_labels_int32'] = prn_labels
    blobs_out['roi_needs_refine_int32'] = roi_needs_refine


def _expand_to_class_specific_prn_targets(prn_labels, class_labels):
    """Expand labels from shape (#rois, ) to (#rois, #classes )
    to encode class specific mask targets.
    """
    assert prn_labels.shape[0] == class_labels.shape[0]

    # Target values of -1 are "don't care" / ignore labels
    prn_targets = -blob_utils.ones(
        (prn_labels.shape[0], cfg.MODEL.NUM_CLASSES), int32=True
    )
    prn_targets[np.arange(prn_labels.shape[0]), class_labels] = prn_labels

    return prn_targets




