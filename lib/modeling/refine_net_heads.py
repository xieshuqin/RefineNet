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

from caffe2.python import core, brew
from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.Hourglass as Hourglass
import utils.blob as blob_utils
import utils.keypoints as keypoint_utils

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
    """ function to determine which type of indicator to use"""
    if cfg.REFINENET.LOCAL_MASK:
        blob_out, dim_out = add_refine_net_local_mask_inputs(
            model, blob_in, dim_in, spatial_scale
        )
    else:
        blob_out, dim_out = add_refine_net_global_mask_inputs(
            model, blob_in, dim_in, spatial_scale
        )

    return blob_out, dim_out


def add_refine_net_local_mask_inputs(model, blob_in, dim_in, spatial_scale):
    if cfg.REFINENET.USE_GPU:
        blob_out, dim_out = add_refine_net_local_mask_inputs_gpu(
            model, blob_in, dim_in, spatial_scale
        )
    else:
        blob_out, dim_out = add_refine_net_local_mask_inputs_cpu(
            model, blob_in, dim_in, spatial_scale
        )
    return blob_out, dim_out


def add_refine_net_local_mask_inputs_gpu(model, blob_in, dim_in, spatial_scale):
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

    # Generate the indicator feature map by
    # 1. up_scale the rois
    # 2. use RoIAlign to pool a M x M feature from the pad_rois,
    #    where M is specified in the cfg.
    # 3. draw the local mask to the pad_rois as an indicator
    # 4. concat the indicator with the pooled feature

    M = cfg.REFINENET.ROI_XFORM_RESOLUTION
    up_scale = cfg.REFINENET.UP_SCALE

    # up_scale mask_rois
    scale_rois = model.ScaleRoIs(
        blob_rois='mask_rois',
        blob_scale_rois='refined_mask_rois',
        up_scale=up_scale
    )
    # use RoIAlign to poor the feature
    rois_global_feat = model.RoIFeatureTransform(
        blob_in,
        blob_out='rois_global_feat',
        blob_rois='refined_mask_rois',
        method='RoIAlign',
        resolution=M,
        sampling_ratio=cfg.REFINENET.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    if cfg.REFINENET.USE_INDICATOR: # whether to use indicator
        # Generate mask indicators
        dim_indicators= cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1
        if cfg.REFINENET.AUTO_LEARNING_INDICATOR:
            # Auto learning indicator
            mask_indicators = model.GenerateAutoLearningIndicators(
                blobs_in='mask_fcn_logits',
                blob_out='mask_indicators',
                blob_rois='mask_rois',
                up_scale=cfg.REFINENET.UP_SCALE,
                resolution=cfg.REFINENET.ROI_XFORM_RESOLUTION
            )
        elif cfg.REFINENET.USE_CUDA_INDICATOR_OP:
            # use indicator op written in cuda
            # Note that this op will run backward to the mask probs, which may
            # not produce the same results as the default python op. To avoid
            # this, we need to add an extra StopGradientOp.
            if cfg.REFINENET.USE_FEATS:
                dim_indicators = cfg.MRCNN.DIM_REDUCED
                mask_indicators = model.GenerateLocalMaskIndicatorsCUDA(
                    blobs_in='conv5_mask',
                    blob_out='mask_indicators',
                    blob_rois='mask_rois',
                    up_scale=cfg.REFINENET.UP_SCALE,
                    resolution=cfg.REFINENET.ROI_XFORM_RESOLUTION,
                )
            else:
                model.net.Sigmoid('mask_fcn_logits', 'mask_probs')
                mask_indicators = model.GenerateLocalMaskIndicatorsCUDA(
                    blobs_in='mask_probs',
                    blob_out='mask_indicators',
                    blob_rois='mask_rois',
                    up_scale=cfg.REFINENET.UP_SCALE,
                    resolution=cfg.REFINENET.ROI_XFORM_RESOLUTION
                )
            if not cfg.REFINENET.BP_TO_INDICATORS:
                # Don't backward to indicators
                mask_indicators = model.net.StopGradient(
                    'mask_indicators', 'mask_indicators'
                )
        elif cfg.REFINENET.GRADIENT_INTO_INDICATOR_ON:
            # Allow gradient to flow from Refinenet to indicators
            mask_probs = model.net.Sigmoid('mask_fcn_logits', 'mask_probs')
            mask_indicators = model.GenerateAutoLearningIndicators(
                blobs_in=mask_probs,
                blob_out='mask_indicators',
                blob_rois='mask_rois',
                up_scale=cfg.REFINENET.UP_SCALE,
                resolution=cfg.REFINENET.ROI_XFORM_RESOLUTION
            )
        else:
            # default setting, use PythonOp
            mask_probs = model.net.Sigmoid('mask_fcn_logits', 'mask_probs')
            blob_data = core.ScopedBlobReference('data')
            mask_indicators = model.GenerateLocalMaskIndicators(
                blobs_in=[blob_data, mask_probs],
                blob_out='mask_indicators',
                blob_rois='mask_rois',
            )

        # Concatenate along the channel dimension
        concat_list = [rois_global_feat, mask_indicators]
        refine_net_input, _ = model.net.Concat(
            concat_list, ['refine_mask_net_input', '_split_info'], axis=1
        )

        blob_out = refine_net_input
        dim_out = dim_in + dim_indicators
    else:
        blob_out = rois_global_feat
        dim_out = dim_in

    if cfg.MODEL.PRN_ON:
        # use IoU Net, sample according to the roi_needs_refine
        blob_out = model.net.SampleAs(
            [blob_out, 'roi_needs_refine_int32'], [blob_out + '_slice']
        )

    return blob_out, dim_out


def add_refine_net_keypoint_inputs(model, blob_in, dim_in, spatial_scale):
    """ Prepare keypoint inputs for RefineNet.
    This function uses keypoint heatmap as indicator and generates input for
    RefineNet. It concantates the indicator with the RoI feature. The resulted
    tensor served as input for RefineNet.
    Input:
        blob_in: FPN/ResNet feature.
        dim_in: FPN/ResNet feature dimension
        spatial_scale: FPN/ResNet scale
    Output:
        'refine_keypoint_net_input'
        dim_out: dim_in + num_cls
    """

    # Generate the indicator feature map by
    # 1. up_scale the rois
    # 2. use RoIAlign to pool a M x M feature from the pad_rois,
    #    where M is specified in the cfg.
    # 3. draw the keypoint heatmap to the pad_rois as an indicator
    # 4. concat the indicator with the pooled feature

    M = cfg.REFINENET.ROI_XFORM_RESOLUTION
    up_scale = cfg.REFINENET.UP_SCALE
    dim_indicators = cfg.KRCNN.NUM_KEYPOINTS

    # up_scale rois
    scale_rois = model.ScaleRoIs(
        blob_rois='keypoint_rois',
        blob_scale_rois='refined_keypoint_rois',
        up_scale=up_scale
    )
    # use RoIAlign to poor the feature
    refined_rois_feat = model.RoIFeatureTransform(
        blob_in,
        blob_out='refined_rois_feat',
        blob_rois='refined_keypoint_rois',
        method='RoIAlign',
        resolution=M,
        sampling_ratio=cfg.REFINENET.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    if cfg.REFINENET.USE_INDICATOR: # whether to use indicator
        # Generate mask indicators
        num_keypoints = cfg.REFINENET.KRCNN.NUM_KEYPOINTS
        blob_data = core.ScopedBlobReference('data')

        if cfg.REFINENET.USE_PROBS_AS_INDICATOR:
            # using probability map for generating keypoint indicator

            # Prepare inputs for PythonOp
            if model.train:
                kps_prob, _ = model.net.Reshape(
                    ['kps_prob'], ['kps_prob_reshaped', '_kps_prob_old_shape'],
                    shape=(-1, cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE)
                )
            else:
                # Test time, we need to generate kps_prob
                model.net.Reshape(
                    ['kps_score'], ['kps_score_reshaped', '_kps_score_old_shape'],
                    shape=(-1, cfg.KRCNN.HEATMAP_SIZE * cfg.KRCNN.HEATMAP_SIZE)
                )
                model.net.Softmax('kps_score_reshaped','kps_prob')
                kps_prob, _ = model.net.Reshape(
                    ['kps_prob'], ['kps_prob_reshaped', '_kps_prob_old_shape'],
                    shape=(-1, cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE)
                )

            # Default setting, just use the local indicator
            kps_indicator = model.GenerateKeypointIndicators(
                blobs_in=[blob_data, kps_prob],
                blob_out='kps_indicator',
                blob_rois='keypoint_rois',
            )
        elif cfg.REFINENET.USE_FEATS:
            # use feature to generate indicators
            dim_indicators = cfg.KRCNN.CONV_HEAD_DIM
            kps_indicator = model.GenerateLocalMaskIndicatorsCUDA(
                blobs_in='conv_fcn8',
                blob_out='kps_indicator',
                blob_rois='keypoint_rois',
                up_scale=cfg.REFINENET.UP_SCALE,
                resolution=cfg.REFINENET.ROI_XFORM_RESOLUTION
            )
        else:
            # use score map as indicator
            kps_score = core.ScopedBlobReference('kps_score')
            kps_indicator = model.GenerateLocalMaskIndicators(
                blobs_in=[blob_data, kps_score],
                blob_out='kps_indicator',
                blob_rois='keypoint_rois'
            )

        # Concatenate along the channel dimension
        concat_list = [refined_rois_feat, kps_indicator]
        refine_net_input, _ = model.net.Concat(
            concat_list, ['refine_keypoint_net_input', '_split_info'], axis=1
        )

        blob_out = refine_net_input
        dim_out = dim_in + dim_indicators
    else:
        blob_out = refined_rois_feat
        dim_out = dim_in

    if cfg.MODEL.PRN_ON:
        # use IoU Net, sample according to the roi_needs_refine
        blob_out = model.net.SampleAs(
            [blob_out, 'roi_needs_refine_int32'], [blob_out + '_slice']
        )

    return blob_out, dim_out


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


def add_refine_net_keypoint_outputs(model, blob_in, dim):
    """Add Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    # NxKxHxW
    upsample_heatmap = (cfg.REFINENET.KRCNN.UP_SCALE > 1)

    if cfg.REFINENET.KRCNN.USE_DECONV:
        # Apply ConvTranspose to the feature representation; results in 2x
        # upsampling
        blob_in = model.ConvTranspose(
            blob_in,
            'refined_kps_deconv',
            dim,
            cfg.REFINENET.KRCNN.DECONV_DIM,
            kernel=cfg.REFINENET.KRCNN.DECONV_KERNEL,
            pad=int(cfg.REFINENET.KRCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        model.Relu('refined_kps_deconv', 'refined_kps_deconv')
        dim = cfg.REFINENET.KRCNN.DECONV_DIM

    if upsample_heatmap:
        blob_name = 'refined_kps_score_lowres'
    else:
        blob_name = 'refined_kps_score'

    if cfg.REFINENET.KRCNN.USE_DECONV_OUTPUT:
        # Use ConvTranspose to predict heatmaps; results in 2x upsampling
        blob_out = model.ConvTranspose(
            blob_in,
            blob_name,
            dim,
            cfg.REFINENET.KRCNN.NUM_KEYPOINTS,
            kernel=cfg.REFINENET.KRCNN.DECONV_KERNEL,
            pad=int(cfg.REFINENET.KRCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.REFINENET.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        # Use Conv to predict heatmaps; does no upsampling
        blob_out = model.Conv(
            blob_in,
            blob_name,
            dim,
            cfg.REFINENET.KRCNN.NUM_KEYPOINTS,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.REFINENET.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

    if upsample_heatmap:
        # Increase heatmap output size via bilinear upsampling
        blob_out = model.BilinearInterpolation(
            blob_out, 'refined_kps_score', cfg.REFINENET.KRCNN.NUM_KEYPOINTS,
            cfg.REFINENET.KRCNN.NUM_KEYPOINTS, cfg.REFINENET.KRCNN.UP_SCALE
        )

    return blob_out

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
    if cfg.MODEL.PIXEL_FOCAL_LOSS_ON:
        # using pixel level focal sigmoid cross entropy loss
        loss_refined_mask = model.net.MaskSigmoidFocalLoss(
            [blob_refined_mask, 'refined_masks_int32'],
            'loss_refined_mask',
            scale=1. / cfg.NUM_GPUS * cfg.REFINENET.WEIGHT_LOSS_MASK,
            gamma=cfg.PIXEL_FOCAL_LOSS.LOSS_GAMMA
        )
    elif cfg.REFINENET.ASSIGN_LARGER_WEIGHT_FOR_CROWDED_SAMPLES:
        loss_refined_mask = model.net.InstanceWeightedSigmoidCrossEntropyLoss(
            [blob_refined_mask, 'refined_masks_int32', 'loss_weights'],
            'loss_refined_mask',
            scale=1. / cfg.NUM_GPUS * cfg.REFINENET.WEIGHT_LOSS_MASK
        )
    else:
        # using normal sigmoid cross entropy loss
        loss_refined_mask = model.net.SigmoidCrossEntropyLoss(
            [blob_refined_mask, 'refined_masks_int32'],
            'loss_refined_mask',
            scale=1. / cfg.NUM_GPUS * cfg.REFINENET.WEIGHT_LOSS_MASK
        )

    loss_gradients = blob_utils.get_loss_gradients(model, [loss_refined_mask])
    model.AddLosses('loss_refined_mask')
    # # And adds MaskIoU ops
    # model.net.Sigmoid(blob_refined_mask, 'refined_mask_probs')
    # model.net.MaskIoU(
    #     ['refined_mask_probs', 'refined_masks_int32'],
    #     ['refined_mask_ious', 'mean_refined_mask_ious']
    # )
    # model.AddMetrics('mean_refined_mask_ious')
    # # And we also want to monitor the mask_ious before refined
    # if cfg.MODEL.PRN_ON:
    #     model.net.SampleAs(
    #         ['mask_ious', 'roi_needs_refine_int32'],
    #         ['prior_mask_ious']
    #     )
    #     model.net.ReduceFrontMean(
    #         'prior_mask_ious',
    #         'mean_prior_mask_ious',
    #         num_reduce_dim=1
    #     )
    #     model.AddMetrics('mean_prior_mask_ious')
    return loss_gradients


def add_refine_net_keypoint_losses(model, blob_refined_keypoint):
    if cfg.MODEL.USE_GAUSSIAN_HEATMAP:
        return add_refine_net_keypoint_losses_gaussian(model, blob_refined_keypoint)
    else:
        return add_refine_net_keypoint_losses_softmax(model, blob_refined_keypoint)


def add_refine_net_keypoint_losses_gaussian(model, blob_refined_keypoint):
    """Add Mask R-CNN keypoint specific losses. Using MSE loss"""
    model.net.Alias(blob_refined_keypoint, 'refined_kps_prob')
    loss_refined_kps = model.net.MeanSquareLoss(
        ['refined_kps_prob', 'refined_keypoint_heatmaps', 'refined_keypoint_weights'], 
        'loss_refined_kps',
        scale=cfg.REFINENET.KRCNN.LOSS_WEIGHT / cfg.NUM_GPUS
    )
    if not cfg.REFINENET.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        model.StopGradient(
            'refined_keypoint_loss_normalizer', 'refined_keypoint_loss_normalizer'
        )
        loss_refined_kps = model.net.Mul(
            ['loss_refined_kps', 'refined_keypoint_loss_normalizer'],
            'loss_refined_kps_normalized'
        )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_refined_kps])
    model.AddLosses(loss_refined_kps)
    return loss_gradients


def add_refine_net_keypoint_losses_softmax(model, blob_refined_keypoint):
    """Add Mask R-CNN keypoint specific losses."""
    # Reshape input from (N, K, H, W) to (NK, HW)
    model.net.Reshape(
        blob_refined_keypoint,
        ['refined_kps_score_reshaped', 'refined_kps_score_old_shape'],
        shape=(-1, cfg.REFINENET.KRCNN.HEATMAP_SIZE * cfg.REFINENET.KRCNN.HEATMAP_SIZE)
    )
    # Softmax across **space** (woahh....space!)
    # Note: this is not what is commonly called "spatial softmax"
    # (i.e., softmax applied along the channel dimension at each spatial
    # location); This is softmax applied over a set of spatial locations (i.e.,
    # each spatial location is a "class").
    refined_kps_prob, loss_refined_kps = model.net.SoftmaxWithLoss(
        ['refined_kps_score_reshaped', 'refined_keypoint_locations_int32',
         'refined_keypoint_weights'],
        ['refined_kps_prob', 'loss_refined_kps'],
        scale=cfg.REFINENET.KRCNN.LOSS_WEIGHT / cfg.NUM_GPUS,
        spatial=0
    )
    if not cfg.REFINENET.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        model.StopGradient(
            'refined_keypoint_loss_normalizer', 'refined_keypoint_loss_normalizer'
        )
        loss_refined_kps = model.net.Mul(
            ['loss_refined_kps', 'refined_keypoint_loss_normalizer'],
            'loss_refined_kps_normalized'
        )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_refined_kps])
    model.AddLosses(loss_refined_kps)
    return loss_gradients


# ---------------------------------------------------------------------------- #
# RefineNet heads
# ---------------------------------------------------------------------------- #
def add_refine_net_head(model, blob_in, dim_in, prefix):
    """
    Function that abstracts away different choices of fcn model.
    Note that the refine head is free of indicator type.
    """
    # note that prefix must be 'mask' or 'keypoint'
    assert prefix in {'mask', 'keypoint'}, \
        'prefix must be mask/keypoints'
    blob_out = 'refine_' + prefix + '_net_feat'
    if cfg.REFINENET.HEAD == 'HOURGLASS':
        n = cfg.REFINENET.NUM_HG_MODULES
        current, dim_inner = Hourglass.add_hourglass_head(
            model, blob_in, 'refined_hg_out', dim_in, prefix, n
        )
        # upsample layer
        model.ConvTranspose(
            current,
            blob_out,
            dim_inner,
            dim_inner,
            kernel=2,
            pad=0,
            stride=2,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
        return blob_out, dim_inner
    elif cfg.REFINENET.HEAD == 'MRCNN_FCN':
        # Use similar heads as Mask head, but changed the names.
        # Note that this head occupies huge GPU memories(~7GB for batch 512).
        num_convs = cfg.REFINENET.MRCNN_FCN.NUM_CONVS
        use_deconv = cfg.REFINENET.MRCNN_FCN.USE_DECONV
        blob_out, dim_out = add_fcn_head(
            model, blob_in, blob_out, dim_in, prefix, num_convs, use_deconv
        )
        return blob_out, dim_out
    elif cfg.REFINENET.HEAD == 'RESNET_FCN':
        # Use resnet-like structures as the head, this should be memory
        # efficiency. (~ 1GB for batch 512)
        n_downsampling = cfg.REFINENET.RESNET_FCN.NUM_DOWNSAMPLING_LAYERS
        num_res_blocks = cfg.REFINENET.RESNET_FCN.NUM_RES_BLOCKS
        use_deconv = cfg.REFINENET.RESNET_FCN.USE_DECONV
        blob_out, dim_out = add_resnet_head(
            model, blob_in, blob_out, dim_in, prefix,
            n_downsampling, num_res_blocks, use_deconv
        )
        return blob_out, dim_out
    elif cfg.REFINENET.HEAD == 'KRCNN':
        # Use keypoint rcnn like head
        blob_out, dim_out = add_krcnn_head(
            model, blob_in, blob_out, dim_in, prefix
        )
        return blob_out, dim_out
    else:
        raise NotImplementedError(
            '{} not supported'.format(cfg.REFINENET.HEAD)
        )


def add_fcn_head(model, blob_in, blob_out, dim_in, prefix, num_convs, use_deconv):

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    current = blob_in
    for i in range(num_convs-1):
        current = model.Conv(
            current,
            prefix+'_[refined_mask]_fcn' + str(i + 1),
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
            prefix+'_[refined_mask]_fcn' + str(num_convs),
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

        model.ConvTranspose(
            current,
            blob_out,
            dim_in,
            dim_inner,
            kernel=2,
            pad=0,
            stride=2,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        model.Conv(
            current,
            blob_out,
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )

    blob_out = model.Relu(blob_out, blob_out)
    dim_out = dim_inner

    return blob_out, dim_out


def add_krcnn_head(model, blob_in, blob_out, dim_in, prefix):
    """Add a Mask R-CNN keypoint head. v1convX design: X * (conv)."""
    hidden_dim = cfg.REFINENET.KRCNN.CONV_HEAD_DIM
    kernel_size = cfg.REFINENET.KRCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2

    current = blob_in
    for i in range(cfg.REFINENET.KRCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'refined_'+ prefix + '_conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.REFINENET.KRCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim


def add_resnet_head(
    model, blob_in, blob_out, dim_in, prefix,
    n_downsampling, num_res_blocks, use_deconv
):
    dilation = cfg.REFINENET.RESNET_FCN.DILATION
    dim_inner = cfg.REFINENET.RESNET_FCN.DIM_REDUCED

    current = blob_in

    # Downsampling
    for i in range(n_downsampling):
        if i > 0:
            dim_inner *= 2
        current = model.Conv(
            current,
            prefix+'_[refined]_resnet_down' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=2,weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # residual blocks
    for i in range(num_res_blocks):
        current = add_residual_block(
            model,
            prefix+'_[refined]_resnet_res' + str(i + 1),
            current,
            dim_in=dim_in,
            dim_out=dim_inner,
            dim_inner=dim_inner,
            dilation=dilation,
            inplace_sum=True
        )
        dim_in = dim_inner

    # Upsampling
    for i in range(n_downsampling):
        if i < n_downsampling - 1:
            dim_inner = int(dim_inner / 2)
        current = model.ConvTranspose(
            current,
            prefix+'_[refined]_resnet_up' + str(n_downsampling - i),
            dim_in=dim_in,
            dim_out=dim_inner,
            kernel=2,
            pad=0,
            out_pad=0,
            stride=2,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
        current = brew.spatial_bn(model, current, current+'_bn', dim_inner, is_test= not model.train)
        current = model.Relu(current, current)

        dim_in = dim_inner

    if use_deconv:
        current = model.Conv(
            current,
            prefix+'_[refined]_resnet_conv' + str(1),
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

        model.ConvTranspose(
            current,
            blob_out,
            dim_in,
            dim_inner,
            kernel=2,
            pad=0,
            stride=2,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        model.Conv(
            current,
            blob_out,
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )

    blob_out = model.Relu(blob_out, blob_out)
    dim_out = dim_inner

    return blob_out, dim_out


def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride=1,
    inplace_sum=False
):
    """Add a residual block to the model."""
    # prefix = res<stage>_<sub_stage>, e.g., res2_3

    # transformation blob
    # conv3x3 -> BN -> Relu
    cur = model.ConvBN(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=3,
        stride=1,
        pad=1 * dilation,
        dilation=dilation
    )
    cur = model.Relu(cur, cur)
    # conv 3x3 -> BN
    tr = model.ConvBN(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_out,
        kernel=3,
        stride=1,
        pad=1 * dilation,
        dilation=dilation
    )

    # shortcut
    sc = blob_in
    # sum -> Relu
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        s = model.net.Sum([tr, sc], prefix + '_sum')

    return model.Relu(s, s)


# ---------------------------------------------------------------------------- #
# Old codes that no longer used
# ---------------------------------------------------------------------------- #
def add_refine_net_local_mask_inputs_cpu(model, blob_in, dim_in, spatial_scale):
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

    # Generate the indicator feature map by
    # 1. up_scale the rois
    # 2. pool the feature from the pad_rois
    # 3. resize to MxM, where M is specified in the cfg

    M = cfg.REFINENET.RESOLUTION
    up_scale = cfg.REFINENET.UP_SCALE
    if isinstance(blob_in, list):
        # FPN case
        rois_global_feat = model.PoolingIndicatorFeatureFPN(
            blobs_in=blob_in,
            blob_out='rois_global_feat',
            blob_rois='mask_rois',
            spatial_scales=spatial_scale
        )
    else:
        #
        rois_global_feat = model.PoolingIndicatorFeatureSingle(
            blobs_in=blob_in,
            blob_out='rois_global_feat',
            blob_rois='mask_rois',
            spatial_scale=spatial_scale
        )


    # Generate mask indicators
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1
    mask_probs = model.net.Sigmoid('mask_fcn_logits', 'mask_probs')
    blob_data = core.ScopedBlobReference('data')
    mask_indicators = model.GenerateLocalMaskIndicators(
        blobs_in=[blob_data, mask_probs],
        blob_out='mask_indicators',
        blob_rois='mask_rois',
    )

    # Concatenate along the channel dimension
    concat_list = [rois_global_feat, mask_indicators]
    refine_net_input, _ = model.net.Concat(
        concat_list, ['refine_mask_net_input', '_split_info'], axis=1
    )

    blob_out = refine_net_input
    dim_out = dim_in + num_cls

    return blob_out, dim_out


def add_refine_net_global_mask_inputs(model, blob_in, dim_in, spatial_scale):
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
    mask_indicators = model.GenerateGlobalMaskIndicators(
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
