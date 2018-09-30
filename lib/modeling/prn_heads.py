"""Various network "heads" for classification.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> prn head -> prn output -> prn loss
... -> Feature /
       Map

The PRN head produces a feature representation of the RoI for the purpose
of classfying whether the roi needs further refinement . The rpn output module
converts the feature representation into classification predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# PRN outputs and losses
# ---------------------------------------------------------------------------- #

def add_prn_outputs(model, blob_in, dim):
    """Add RoI classification output ops."""
    blob_out = model.FC(
        blob_in,
        'prn_logits',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add sigmoid when testing; during training the sigmoid is
        # combined with the label cross entropy loss for numerical stability
        blob_out = model.net.Sigmoid('prn_logits', 'prn_probs', engine='CUDNN')

    return blob_out

def add_prn_losses(model):
    """Add losses for RoI classification."""
    loss_prn = model.net.SigmoidCrossEntropyLoss(
        ['prn_logits', 'prn_labels_int32'],
        'loss_prn',
        scale=1. / cfg.NUM_GPUS
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_prn])
    model.AddLosses(['loss_prn'])
    # And add some useful metrics
    model.net.Sigmoid('prn_logits', 'prn_probs')
    model.SigmoidAccuracy(['prn_probs', 'prn_labels_int32'], 'accuracy_prn')
    model.AddMetrics('accuracy_prn')
    model.AddMetrics('refine_ratio')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# PRN heads
# ---------------------------------------------------------------------------- #

def add_prn_head(model, blob_in, dim_in, spatial_scale, prefix):
    return add_roi_2mlp_head(
        model, blob_in, dim_in, spatial_scale, prefix
    )

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale, prefix):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.PRN.MLP_HEAD_DIM
    roi_size = cfg.PRN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'prn_roi_feat',
        blob_rois=prefix + '_rois',
        method=cfg.PRN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.PRN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'prn_fc1', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('prn_fc1', 'prn_fc1')
    model.FC('prn_fc1', 'prn_fc2', hidden_dim, hidden_dim)
    model.Relu('prn_fc2', 'prn_fc2')
    return 'prn_fc2', hidden_dim


# ---------------------------------------------------------------------------- #
# PRN labels
# ---------------------------------------------------------------------------- #

def add_prn_labels(model):
    assert model.train, 'Only valid at training'
    model.PrepareLabelsForPRNAndUpdateRefineBlobs()
