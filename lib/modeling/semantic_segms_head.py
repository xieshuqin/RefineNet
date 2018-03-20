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

"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.ResNet as ResNet
import utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Semantic Segmentation outputs and losses
# ---------------------------------------------------------------------------- #

def add_semantic_segms_outputs(model, blob_in, dim):
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    num_cls = cfg.MODEL.NUM_CLASSES 

    # Predict mask using Conv

    # Use GaussianFill for class-agnostic mask prediction; fills based on
    # fan-in can be too large in this case and cause divergence
    fill = (
        cfg.MRCNN.CONV_INIT
        if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
    )
    blob_out = model.Conv(
        blob_in,
        'semantic_segms_fcn_logits',
        dim,
        num_cls,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(fill, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out


def add_semantic_segms_losses(model, blob_semantic_segms):
    """Add Mask R-CNN specific losses."""
    loss_segmantic_segms = model.net.SigmoidCrossEntropyLoss(
        [blob_semantic_segms, 'semantic_segms_int32'],
        'loss_segmantic_segms',
        scale=1. / cfg.NUM_GPUS * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_segmantic_segms])
    model.AddLosses('loss_segmantic_segms')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Semantic Segmentation heads
# ---------------------------------------------------------------------------- #

def add_semantic_segms_head(model, blob_in, dim_in):
    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    num_convs = cfg.SEMANTIC_NET.NUM_CONVS
    use_deconv = cfg.SEMANTIC_NET.USE_DECONV

    current = blob_in
    for i in range(num_convs-1):
        current = model.Conv(
            current,
            'semantic_segms_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    if use_deconv:
        current = model.Conv(
            current,
            'semantic_segms_fcn' + str(num_convs),
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        # upsample layer
        current = model.ConvTranspose(
            current,
            'semantic_segms_feature',
            dim_inner,
            dim_inner,
            kernel=2,
            pad=0,
            stride=2,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        current = model.Conv(
            current,
            'semantic_segms_feature',
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )

    blob_mask = model.Relu(current, current)
    return blob_mask, dim_inner
