from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
import roi_data.prn
import utils.blob as blob_utils

class PrepareLabelsForPRNAndUpdateRefineBlobsOp(object):
    """ Prepare labels for PRN. And also update labels for the refinement
        tasks.

        inputs is [mask_ious, labels_int32]
        if training, then inputs include labels for refinement task,
        such as [refined_masks_int32]

        outputs is [prn_labels_int32, roi_needs_refine_int32, refine_ratio] 
        and also includes labels for refinement task, such as 
        [refined_masks_int32]
    """
    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        # prepare blobs_in
        blobs_in = convert_inputs_to_dict(inputs)
        # prepare blobs_out
        blob_out_names = get_op_blob_out_names()
        blobs_out = {k: [] for k in blob_out_names}
        # add blobs for prn
        roi_data.prn.add_prn_blobs(blobs_out, blobs_in)
        # update refine blobs
        update_refine_blobs(blobs_out, blobs_in)
        # add to outputs
        for i, k in enumerate(blob_out_names):
            blob_utils.py_op_copy_blob(blobs_out[k], outputs[i])


def convert_inputs_to_dict(inputs):
    blobs_in_names = get_op_blob_in_names()
    blobs_in = {k: [] for k in blobs_in_names}
    for i, k in enumerate(blobs_in_names):
        blobs_in[k] = inputs[i].data

    return blobs_in

def get_refine_blob_names():
    blob_names = []
    if cfg.MODEL.REFINE_MASK_ON:
        blob_names += ['refined_masks_int32']
    return blob_names

def get_op_blob_in_names():
    blob_names = ['mask_ious', 'labels_int32']
    blob_names += get_refine_blob_names()
    return blob_names

def get_op_blob_out_names():
    blob_names = roi_data.prn.get_prn_blob_names()
    blob_names += get_refine_blob_names()
    return blob_names

def update_refine_blobs(blobs_out, blobs_in):
    # convert roi_needs_refine_int32 to bool
    roi_needs_refine = blobs_out['roi_needs_refine_int32'].astype(np.bool)
    # update the refine blobs
    blob_names = get_refine_blob_names()
    for k in blob_names:
        blobs_out[k] = blobs_in[k][roi_needs_refine]




