"""Various network "heads" for refining prediction in RefineNet.

The design is as follows:

... -> RoI   -\
               -> Indicator -\
... -> Local -/               \
       Output                  -> refine input -> refine head -> refine output -> loss
                              /
... ----------> Feature Map -/

The local output is the prediction of mask/keypoint head. The RoI is then used
for mapping the local output into the original image space. Then it concantated
with the entire feature map, generating input for RefineNet head. The RefineNet
then generates a new refined prediction (soft masks/ keypoint heatmap)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.Hourglass as Hourglass
import utils.blob as blob_utils

# ---------------------------------------------------------------------------- #
# RefineNet input
# ---------------------------------------------------------------------------- #
def add_refine_net_inputs(model, blob_in, dim_in, spatial_scale, indicator_type):
    """ Specify indicator type and generates different inputs """
    assert indicator_type in {'Mask', 'Keypoint'}, \
        'Only Mask/Keypoint are supported for indicator'
    if indicator_type == 'Mask':
        blob_out, dim_out = add_refine_net_mask_inputs(
            model, blob_in, dim_in, spatial_scale
        )
    else:
        blob_out, dim_out = add_refine_net_keypoint_inputs(
            model, blob_in, dim_in, spatial_scale
        )
    return blob_out, dim_out


def add_refine_net_mask_inputs(model, blob_in, dim_in, spatial_scale):
    """ Prepare mask inputs for RefineNet.
    This function uses mask as indicator and generates input for
    RefineNet. It maps the local mask prediction to global image
    space, which serves as an indicator, and concantate the
    indicator with the entire feature map. The resulted tensor
    served as input for RefineNet.
    Input:
        blob_in: FPN/ResNet feature.
        dim_in: FPN/ResNet feature dimension
        spatial_scale: FPN/ResNet scale
    Output:
        'refine_mask_net_input'
        dim_out: dim_in + num_cls
    """

    # Rescale and Dumplicate the feature map
    src_sc = spatial_scale
    dst_sc = cfg.REFINENET.SPATIAL_SCALE

    if cfg.FPN.FPN_ON: 
        rois_global_feat = model.RescaleAndDumplicateFeatureFPN(
            blobs_in=blob_in,
            blob_out='rois_global_feat',
            blob_rois='mask_rois',
            src_spatial_scales=src_sc,
            dst_spatial_scale=dst_sc
        )
    else:
        rois_global_feat = model.RescaleAndDumplicateFeatureSingle(
            blobs_in=blob_in,
            blob_out='rois_global_feat',
            blob_rois='mask_rois',
            src_spatial_scales=src_sc,
            dst_spatial_scale=dst_sc
        )


    # Generate mask indicators
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1
    mask_probs = model.net.Sigmoid('mask_fcn_logits', 'mask_probs')
    blob_data = core.ScopedBlobReference('data')
    mask_indicators = model.GenerateMaskIndicators(
        blobs_in=[blob_data, mask_probs],
        blob_out='mask_indicators',
        blob_rois='mask_rois',
        dst_spatial_scale=dst_sc
    )

    # Concatenate along the channel dimension
    concat_list = [rois_global_feat, mask_indicators]
    refine_net_input, _ = model.net.Concat(
        concat_list, ['refine_mask_net_input', '_split_info'], axis=1
    )

    blob_out = refine_net_input
    dim_out = dim_in + num_cls

    return blob_out, dim_out


def add_refine_net_keypoint_inputs(model, blob_in, dim_in, spatial_scale):
    pass


# ---------------------------------------------------------------------------- #
# RefineNet outputs
# ---------------------------------------------------------------------------- #
def add_refine_net_outputs(model, blob_in, dim_in, refined_output_type):
    """ Specify refined output type """
    assert refined_output_type in {'Mask', 'Keypoint'}, \
        'Only Mask/Keypoint are supported as refined output type'
    if refined_output_type == 'Mask':
        blob_out = add_refine_net_mask_outputs(model, blob_in, dim_in)
    else:
        blob_out = add_refine_net_keypoint_outputs(model, blob_in, dim_in)

    return blob_out


def add_refine_net_mask_outputs(model, blob_in, dim_in):
    """ add Refine Net output
    blob_in: 'refine_mask_net_feat'
    blob_out: 'refined_mask_logits' or 'refined_mask_probs'
    """
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1

    # Use GaussianFill for class-agnostic mask prediction; fills based on
    # fan-in can be too large in this case and cause divergence
    fill = (
        cfg.MRCNN.CONV_INIT
        if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
    )
    blob_out = model.Conv(
        blob_in,
        'refined_mask_logits',
        dim_in,
        num_cls,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(fill, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'refined_mask_probs')

    return blob_out


def add_refine_net_keypoint_outputs(model, blob_in, dim_in):
    pass


# ---------------------------------------------------------------------------- #
# RefineNet losses
# ---------------------------------------------------------------------------- #
def add_refine_net_losses(model, blob_refined, refined_output_type):
    """ Specify output type loss """
    assert refined_output_type in {'Mask', 'Keypoint'}, \
        'Only Mask/Keypoint are supported for refined output type'
    if refined_output_type == 'Mask':
        loss_gradients = add_refine_net_mask_losses(model, blob_refined)
    else:
        loss_gradients = add_refine_net_keypoint_losses(model, blob_refined)

    return loss_gradients


def add_refine_net_mask_losses(model, blob_refined_mask):
    """ Add RefineNet mask specific losses. """
    loss_refined_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_refined_mask, 'refined_masks_int32'],
        'loss_refined_mask',
        scale=1. / cfg.NUM_GPUS * cfg.REFINENET.WEIGHT_LOSS_MASK
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_refined_mask])
    model.AddLosses('loss_refined_mask')
    return loss_gradients


def add_refine_net_keypoint_losses(model, blob_refined_keypoint):
    pass


# ---------------------------------------------------------------------------- #
# RefineNet heads
# ---------------------------------------------------------------------------- #
def add_refine_net_head(model, blob_in, dim_in, prefix):
    """ Currently, we only consider using Hourglass as the RefineNet head,
    however it can be expanded to other types of network. Therefore we
    will leave a function to allow different choices of fcn model.
    Note that the refine head is free of indicator type.
    """
    # note that prefix must be 'mask' or 'keypoint'
    assert prefix in {'mask', 'keypoints'}, \
        'prefix must be mask/keypoints'
    blob_out = 'refine_' + prefix + '_net_feat'
    if cfg.REFINENET.HEAD == 'HOURGLASS':
        blob_out, dim_out = Hourglass.add_hourglass_head(
            model, blob_in, blob_out, dim_in, prefix
        )
        return blob_out, dim_out

