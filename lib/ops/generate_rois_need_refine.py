from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
import utils.blob as blob_utils

class GenerateRoIsNeedRefineOp(object):
    def __init__(self, train):
        self._train = train

    def forward(self, inputs, outputs):
        """ See modeling.detector.GenerateRoIsNeedRefine for
        inputs/outputs documentation.
        """
        # inputs is 
        # [prn_probs]
        # if training, then inputs include [prn_labels_int32]
        # if training, then inputs include labels for refinement task,
        # such as [refined_masks_int32]
        probs = inputs[0].data
        if self._train:
            blob_names = get_output_blob_names(self._train)
            blobs = {k: [] for k in blob_names}
            # use prn_labels_int32 as the output 
            labels = inputs[1].data
            assert labels.dtype == np.int32, 'labels type must be int32'
            blobs['roi_needs_refine_int32'] = labels
            # use the label to slice other inputs
            # and then overwrite the original inputs
            slice_blobs(inputs[2:], labels, blobs, blob_names[1:])
            for i, k in enumerate(blob_names):
                blob_utils.py_op_copy_blob(blobs[k], outputs[i])
        else:
            # testing time, binarize the probs and decides the indexes
            blob_out = (probs > 0.5).astype(np.int32, copy=False)
            blob_utils.py_op_copy_blob(blob_out, outputs[0])


def get_output_blob_names(is_training=True):
    blob_names = ['roi_needs_refine_int32']
    if is_training:
        if cfg.MODEL.REFINE_MASK_ON:
            blob_names += ['refined_masks_int32']

    return blob_names

def slice_blobs(inputs, index, blobs_out, blob_out_names):
    # slice the inputs
    assert len(inputs) == len(blob_out_names), 'Length not equal'
    for i in range(len(inputs)):
        data = inputs[i].data
        assert index.shape[0] == data.shape[0], 'Shape not equal'

        name = blob_out_names[i]
        blobs_out[name] = data[index].astype(data.dtype)


