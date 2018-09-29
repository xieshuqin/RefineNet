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

"""Defines DetectionModelHelper, the class that represents a Detectron model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging

from caffe2.python import cnn
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python import brew

from core.config import cfg
from ops.collect_and_distribute_fpn_rpn_proposals \
    import CollectAndDistributeFpnRpnProposalsOp
from ops.generate_proposal_labels import GenerateProposalLabelsOp
from ops.generate_proposals import GenerateProposalsOp

from ops.rescale_and_dumplicate_feature \
    import RescaleAndDumplicateFeatureSingleOp

from ops.rescale_and_dumplicate_feature \
    import RescaleAndDumplicateFeatureFPNOp

from ops.pooling_feature import PoolingIndicatorFeatureSingleOp
from ops.pooling_feature import PoolingIndicatorFeatureFPNOp

from ops.scale_rois import ScaleRoIsSingleOp, ScaleRoIsFPNOp
from ops.prepare_labels_for_prn_and_update_refine_blobs import \
    PrepareLabelsForPRNAndUpdateRefineBlobsOp
from ops.generate_rois_need_refine import GenerateRoIsNeedRefineOp

from ops.generate_mask_indicators import GenerateGlobalMaskIndicatorsOp
from ops.generate_mask_indicators import GenerateLocalMaskIndicatorsOp
from utils import lr_policy
import roi_data.fast_rcnn
import ops.prepare_labels_for_prn_and_update_refine_blobs as prn_label_op
import utils.c2 as c2_utils

logger = logging.getLogger(__name__)


class DetectionModelHelper(cnn.CNNModelHelper):
    def __init__(self, **kwargs):
        # Handle args specific to the DetectionModelHelper, others pass through
        # to CNNModelHelper
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'num_classes must be > 0'
        for k in ('train', 'num_classes'):
            if k in kwargs:
                del kwargs[k]
        kwargs['order'] = 'NCHW'
        # Defensively set cudnn_exhaustive_search to False in case the default
        # changes in CNNModelHelper. The detection code uses variable size
        # inputs that might not play nicely with cudnn_exhaustive_search.
        kwargs['cudnn_exhaustive_search'] = False
        super(DetectionModelHelper, self).__init__(**kwargs)
        self.roi_data_loader = None
        self.losses = []
        self.metrics = []
        self.do_not_update_params = []  # Param on this list are not updated
        self.net.Proto().type = cfg.MODEL.EXECUTION_TYPE
        self.net.Proto().num_workers = cfg.NUM_GPUS * 4
        self.prev_use_cudnn = self.use_cudnn

    def TrainableParams(self, gpu_id=-1):
        """Get the blob names for all trainable parameters, possibly filtered by
        GPU id.
        """
        return [
            p for p in self.params
            if (
                p in self.param_to_grad and   # p has a gradient
                p not in self.do_not_update_params and  # not on the blacklist
                (gpu_id == -1 or  # filter for gpu assignment, if gpu_id set
                 str(p).find('gpu_{}'.format(gpu_id)) == 0)
            )]

    def AffineChannel(self, blob_in, blob_out, share_with=None, inplace=False):
        """Affine transformation to replace BN in networks where BN cannot be
        used (e.g., because the minibatch size is too small).

        The AffineChannel parameters may be shared with another AffineChannelOp
        by specifying its blob name (excluding the '_{s,b}' suffix) in the
        share_with argument. The operations can be done in place to save memory.
        """
        blob_out = blob_out or self.net.NextName()
        is_not_sharing = share_with is None
        param_prefix = blob_out if is_not_sharing else share_with
        scale = core.ScopedBlobReference(
            param_prefix + '_s', self.param_init_net)
        bias = core.ScopedBlobReference(
            param_prefix + '_b', self.param_init_net)
        if is_not_sharing:
            self.net.Proto().external_input.extend([str(scale), str(bias)])
            self.params.extend([scale, bias])
            self.weights.append(scale)
            self.biases.append(bias)
        if inplace:
            return self.net.AffineChannel([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineChannel([blob_in, scale, bias], blob_out)

    def GenerateProposals(self, blobs_in, blobs_out, anchors, spatial_scale):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        name = 'GenerateProposalsOp:' + ','.join([str(b) for b in blobs_in])
        self.net.Python(
            GenerateProposalsOp(anchors, spatial_scale, self.train).forward
        )(blobs_in, blobs_out, name=name)
        return blobs_out

    def GenerateProposalLabels(self, blobs_in):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        name = 'GenerateProposalLabelsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # The list of blobs is not known before run-time because it depends on
        # the specific model being trained. Query the data loader to get the
        # list of output blob names.
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        self.net.Python(GenerateProposalLabelsOp().forward)(
            blobs_in, blobs_out, name=name
        )
        return blobs_out

    def CollectAndDistributeFpnRpnProposals(self):
        """Merge RPN proposals generated at multiple FPN levels and then
        distribute those proposals to their appropriate FPN levels. An anchor
        at one FPN level may predict an RoI that will map to another level,
        hence the need to redistribute the proposals.

        This function assumes standard blob names for input and output blobs.

        Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                      rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
          - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
            documentation from GenerateProposals.
          - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
            level i; see rpn_roi_probs documentation from GenerateProposals.

        If used during training, then the input blobs will also include:
          [roidb, im_info] (see GenerateProposalLabels).

        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
          - rois_fpn<i> are the RPN proposals for FPN level i
          - rois_idx_restore is a permutation on the concatenation of all
            rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
            restored to their original order in the input blobs.

        If used during training, then the output blobs will also include:
          [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
        """
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL

        # Prepare input blobs
        rois_names = ['rpn_rois_fpn' + str(l) for l in range(k_min, k_max + 1)]
        score_names = [
            'rpn_roi_probs_fpn' + str(l) for l in range(k_min, k_max + 1)
        ]
        blobs_in = rois_names + score_names
        if cfg.MODEL.REFINE_MASK_ON or cfg.MODEL.REFINE_KEYPOINTS_ON:
            blobs_in += ['data']
        if self.train:
            blobs_in += ['roidb', 'im_info']
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'CollectAndDistributeFpnRpnProposalsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        outputs = self.net.Python(
            CollectAndDistributeFpnRpnProposalsOp(self.train).forward
        )(blobs_in, blobs_out, name=name)

        return outputs

    def DropoutIfTraining(self, blob_in, dropout_rate):
        """Add dropout to blob_in if the model is in training mode and
        dropout_rate is > 0."""
        blob_out = blob_in
        if self.train and dropout_rate > 0:
            blob_out = self.Dropout(
                blob_in, blob_in, ratio=dropout_rate, is_test=False
            )
        return blob_out

# ---------------------------------------------------------------------------- #
# Beginning of shuqin's code
# ---------------------------------------------------------------------------- #
    def ScaleRoIs(self, blob_rois, blob_scale_rois, up_scale):
        """ Scale the blob_rois by a up_scale factor.
            abstract the use of FPN here.
        """
        if cfg.FPN.FPN_ON:
            # FPN case
            k_max = cfg.FPN.ROI_MAX_LEVEL
            k_min = cfg.FPN.ROI_MIN_LEVEL
            blobs_in_list = ['data']
            blobs_out_list = []
            for lvl in range(k_min, k_max+1):
                blob_roi = blob_rois + '_fpn' + str(lvl)
                blob_scale_roi = blob_scale_rois + '_fpn' + str(lvl)
                blobs_in_list.append(blob_roi)
                blobs_out_list.append(blob_scale_roi)

            # add the *_idx_restore_int32 to the blobs_list
            restore_bl = blob_rois + '_idx_restore_int32'
            scale_restore_bl = blob_scale_rois + '_idx_restore_int32'
            blobs_in_list.append(restore_bl)
            blobs_out_list.append(scale_restore_bl)
            # Scoped the blob names
            blobs_in_list = [core.ScopedBlobReference(b) for b in blobs_in_list]
            blobs_out_list = [core.ScopedBlobReference(b) for b in blobs_out_list]
            name = 'ScaleRoIsFPNOp: ' + ','.join(
                [str(b) for b in blobs_in_list]
            )

            xform_out = self.net.Python(
                ScaleRoIsFPNOp(k_min, k_max, up_scale).forward
            )(blobs_in_list, blobs_out_list, name=name)

        else:
            # Single RPN case
            blob_rois = core.ScopedBlobReference(blob_rois)
            blob_scale_rois = core.ScopedBlobReference(blob_scale_rois)
            name = 'ScaleRoIsSingleOp: ' + str(blob_rois)

            xform_out = self.net.Python(
                ScaleRoIsSingleOp(up_scale).forward
            )(blob_rois, blob_scale_rois, name=name)

        return xform_out

    def GenerateLocalMaskIndicators(
        self,
        blobs_in,
        blob_out,
        blob_rois='mask_rois',
    ):
        """ Add mask indicators to the refine network. It maps the
        'mask_probs' into the input images' space, and narrow it down
        by the value 'scale'

        Input blobs: [data, mask_probs]
        Input rois: mask_rois
        Output blob: mask_indicators
        """
        blob_rois = core.ScopedBlobReference(blob_rois) # refer blob_rois
        blobs_in_list = blobs_in + [blob_rois]
        name = 'GenerateMaskIndicatorsOp:' + ','.join(
            [str(b) for b in blobs_in_list]
        )
        blob_out = core.ScopedBlobReference(blob_out)
        grad_input_indices=[0] # ignore gradient for blob_rois

        up_scale = cfg.REFINENET.UP_SCALE
        M = cfg.REFINENET.ROI_XFORM_RESOLUTION

        xform_out = self.net.Python(
            GenerateLocalMaskIndicatorsOp(up_scale=up_scale, resolution=M).forward,
            GenerateLocalMaskIndicatorsOp(up_scale=up_scale, resolution=M).backward,
            grad_input_indices=grad_input_indices
        )(blobs_in_list, blob_out, name=name)
        return xform_out

    def PrepareLabelsForPRNAndUpdateRefineBlobs(self):
        """ Prepare labels for PRN and update blobs for RefineNet.

        Input blobs: ['mask_ious', 'labels_int32']
          - labels_int32 is the cls label for rois

        If used during training, adds the related blobs for the specific
        refinement task, such as ['refined_masks_int32'].

        Output blob: ['prn_labels_int32', 'roi_needs_refine_int32']
          - prn_labels_int32 is the labels for prn.
          - roi_needs_refine_int32 is a binary label indicates whether
          further refinement is needed or not.

        And if used during training, also doing inplace-update for the
        labels of refinement tasks. Such as update ['refined_masks_int32']
        """
        # Prepare input blobs
        blobs_in = prn_label_op.get_op_blob_in_names()
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'PrepareLabelsForPRNAndUpdateRefineBlobsOp: ' + ','.join([str(b) for b in blobs_in])
        # Prepare output blobs
        blobs_out = prn_label_op.get_op_blob_out_names()
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]
        # Execute op
        output = self.net.Python(
            PrepareLabelsForPRNAndUpdateRefineBlobsOp().forward
        )(blobs_in, blobs_out, name=name)
        return output

    def GenerateRoIsNeedRefine(self):
        ### IMPORTANT! Unused op!!!

        """ Generate a binary label to decide whether the prediction needs
        further refinement. And also update the corresponding prediction
        here.

        Input blobs: ['prn_probs']
          - prn_probs is the probability of the mask/keypoint
          prediction needs further refinement

        If used during training, includes the labels for PredictNeedRefine
        and use it as the output. ['prn_labels_int32']
        And also adds the related blobs for the specific refinement task,
        such as ['refined_masks_int32'].

        Output blob: ['roi_needs_refine_int32']
          - roi_needs_refine_int32 is a binary label indicates whether
          further refinement is needed or not.

        And if used during training, also doing inplace-update for the
        labels of refinement tasks. Such as update ['refined_masks_int32']
        """
        # Prepare input blobs
        blobs_in = ['prn_probs']
        if self.train:
            # adds labels of prn
            blobs_in += ['prn_labels_int32']
            # adds refinement tasks specific blobs
            if cfg.MODEL.REFINE_MASK_ON:
                blobs_in += ['refined_masks_int32']

        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'GenerateRoIsNeedRefineOp: ' + ','.join(
            str(b) for b in blobs_in
        )

        # Prepare output blobs
        blobs_out = ['roi_needs_refine_int32']
        if self.train:
            # add refinement tasks specific blobs
            if cfg.MODEL.REFINE_MASK_ON:
                blobs_out += ['refined_masks_int32']

        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        # Execute op
        # Note that this op will overwrite the label for the specific task
        outputs = self.net.Python(
            GenerateRoIsNeedRefineOp(self.train).forward
        )(blobs_in, blobs_out, name=name)

        return outputs[0] # only return the binary label

    def MaskIoUs(self, blobs_in, blob_label, blob_out):
        """ Calculate Mask IoUs.
            Input blobs: ['mask_probs', 'masks_int32']
            Output blobs: ['mask_ious']
        """
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'MaskIoUsOp: ' + ','.join([str(b) for b in blobs_in])

        blob_out = core.ScopedBlobReference(blob_out)

        output = self.net.Python(
            MaskIoUsOp().forward
        )(blobs_in, blob_out, name=name)
        return output

# ---------------------------------------------------------------------------- #
# End of shuqin's code
# ---------------------------------------------------------------------------- #
    def RoIFeatureTransform(
        self,
        blobs_in,
        blob_out,
        blob_rois='rois',
        method='RoIPoolF',
        resolution=7,
        spatial_scale=1. / 16.,
        sampling_ratio=0
    ):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)
        has_argmax = (method == 'RoIPoolF')
        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                bl_out = blob_out + '_fpn' + str(lvl)
                bl_out_list.append(bl_out)
                bl_argmax = ['_argmax_' + bl_out] if has_argmax else []
                self.net.__getattr__(method)(
                    [bl_in, bl_rois], [bl_out] + bl_argmax,
                    pooled_w=resolution,
                    pooled_h=resolution,
                    spatial_scale=sc,
                    sampling_ratio=sampling_ratio
                )
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled, _ = self.net.Concat(
                bl_out_list, [blob_out + '_shuffled', '_concat_' + blob_out],
                axis=0
            )
            # Unshuffle to match rois from dataloader
            restore_bl = blob_rois + '_idx_restore_int32'
            xform_out = self.net.BatchPermutation(
                [xform_shuffled, restore_bl], blob_out
            )
        else:
            # Single feature level
            bl_argmax = ['_argmax_' + blob_out] if has_argmax else []
            # sampling_ratio is ignored for RoIPoolF
            xform_out = self.net.__getattr__(method)(
                [blobs_in, blob_rois], [blob_out] + bl_argmax,
                pooled_w=resolution,
                pooled_h=resolution,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio
            )
        # Only return the first blob (the transformed features)
        return xform_out

    def ConvShared(
        self,
        blob_in,
        blob_out,
        dim_in,
        dim_out,
        kernel,
        weight=None,
        bias=None,
        **kwargs
    ):
        """Add conv op that shares weights and/or biases with another conv op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if use_bias:
            blobs_in = [blob_in, weight, bias]
        else:
            blobs_in = [blob_in, weight]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        return self.net.Conv(
            blobs_in, blob_out, kernel=kernel, order=self.order, **kwargs
        )

    def BilinearInterpolation(
        self, blob_in, blob_out, dim_in, dim_out, up_scale
    ):
        """Bilinear interpolation in space of scale.

        Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

        Adapted from the CVPR'15 FCN code.
        See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        """
        assert dim_in == dim_out
        assert up_scale % 2 == 0, 'Scale should be even'

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (dim_in, dim_out, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(dim_out), range(dim_in), :, :] = bil_filt

        blob = self.ConvTranspose(
            blob_in,
            blob_out,
            dim_in,
            dim_out,
            kernel_size,
            stride=int(up_scale),
            pad=int(up_scale / 2),
            weight_init=('GivenTensorFill', {'values': kernel}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        self.do_not_update_params.append(self.weights[-1])
        self.do_not_update_params.append(self.biases[-1])
        return blob

    def ConvAffine(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn',
        inplace=False
    ):
        """ConvAffine adds a Conv op followed by a AffineChannel op (which
        replaces BN during fine tuning).
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1
        )
        blob_out = self.AffineChannel(
            conv_blob, prefix + suffix, inplace=inplace
        )
        return blob_out

    def ConvBN(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn'
    ):
        """ ConvBN adds a Conv op followed by a SpatialBN op. """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1
        )
        blob_out = brew.spatial_bn(
            self, conv_blob, prefix + suffix, dim_out, is_test= not self.train
        )
        return blob_out

    def DisableCudnn(self):
        self.prev_use_cudnn = self.use_cudnn
        self.use_cudnn = False

    def RestorePreviousUseCudnn(self):
        prev_use_cudnn = self.use_cudnn
        self.use_cudnn = self.prev_use_cudnn
        self.prev_use_cudnn = prev_use_cudnn

    def UpdateWorkspaceLr(self, cur_iter):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        # The workspace is the one source of truth for the lr
        # The lr is always the same on all GPUs
        cur_lr = workspace.FetchBlob('gpu_0/lr')[0]
        new_lr = lr_policy.get_lr_at_iter(cur_iter)
        # There are no type conversions between the lr in Python and the lr in
        # the GPU (both are float32), so exact comparision is ok
        if cur_lr != new_lr:
            ratio = _get_lr_change_ratio(cur_lr, new_lr)
            if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
                logger.info(
                    'Changing learning rate {:.6f} -> {:.6f} at iter {:d}'.
                    format(cur_lr, new_lr, cur_iter))
            self._SetNewLr(cur_lr, new_lr)
        return new_lr

    def _SetNewLr(self, cur_lr, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        for i in range(cfg.NUM_GPUS):
            with c2_utils.CudaScope(i):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array([new_lr], dtype=np.float32))
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            self._CorrectMomentum(new_lr / cur_lr)

    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is
        the stochastic gradient. Since V is not defined independently of the
        learning rate (as it should ideally be), when the learning rate is
        changed we should scale the update history V in order to make it
        compatible in scale with lr * grad.
        """
        logger.info(
            'Scaling update history by {:.6f} (new lr / old lr)'.
            format(correction))
        for i in range(cfg.NUM_GPUS):
            with c2_utils.CudaScope(i):
                for param in self.TrainableParams(gpu_id=i):
                    op = core.CreateOperator(
                        'Scale', [param + '_momentum'], [param + '_momentum'],
                        scale=correction)
                    workspace.RunOperatorOnce(op)

    def AddLosses(self, losses):
        if not isinstance(losses, list):
            losses = [losses]
        # Conversion to str allows losses to include BlobReferences
        losses = [c2_utils.UnscopeName(str(l)) for l in losses]
        self.losses = list(set(self.losses + losses))

    def AddMetrics(self, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = list(set(self.metrics + metrics))

# ---------------------------------------------------------------------------- #
# Old codes that no longer used
# ---------------------------------------------------------------------------- #
    def RescaleAndDumplicateFeatureFPN(
        self,
        blobs_in,
        blob_out,
        blob_rois,
        src_spatial_scales,
        dst_spatial_scale
    ):
        """ Dumplicate FPN feature maps for the refiner network.
        If use FPN, then call. Then concancate the feature maps
        along the batch dimension

        Input blobs: [fpn_<min>, ..., fpn_<max>]
        Input rois: [mask_rois_fpn<min>, ..., mask_rois_fpn<max>]

        Output blobs: rois_global_feature
        """
        dst_sc = dst_spatial_scale

        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        blob_fpn_rois = [
            core.ScopedBlobReference(blob_rois+'_fpn'+str(lvl))
            for lvl in range(k_min, k_max+1)
        ]

        src_sc = []
        blobs_in_list = []
        for lvl in range(k_min, k_max+1):
            blob_in = blobs_in[k_max - lvl] # reversed order
            src_sc.append(src_spatial_scales[k_max - lvl]) # reversed order
            blob_fpn_roi = blob_fpn_rois[lvl - k_min]
            blobs_in_list.append(blob_in)
            blobs_in_list.append(blob_fpn_roi)

        name = 'RescaleAndDumplcateFeatureFPNOp: ' + ','.join(
                [str(b) for b in blobs_in_list]
            )
        # ignore gradient for 'blob_rois'
        grad_input_indices = [2*(i-k_min) for i in range(k_min, k_max+1)]
        #grad_input_indices=[]

        blob_fpn_dumplicate_out = [
            core.ScopedBlobReference(blob_out+'_fpn'+str(lvl))
            for lvl in range(k_min, k_max+1)
        ]

        #Rescale and Dumplicate FPN feature
        blob_dumplicate_list = self.net.Python(
            RescaleAndDumplicateFeatureFPNOp(k_min,k_max,src_sc,dst_sc).forward,
            RescaleAndDumplicateFeatureFPNOp(k_min,k_max,src_sc,dst_sc).backward,
            grad_input_indices=grad_input_indices
        )(blobs_in_list, blob_fpn_dumplicate_out, name=name)

        # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
        xform_shuffled, _ = self.net.Concat(
            blob_dumplicate_list, [blob_out + '_shuffled', '_concat_' + blob_out],
            axis=0
        )
        # Unshuffle to match rois from dataloader
        restore_bl = core.ScopedBlobReference(blob_rois + '_idx_restore_int32')
        xform_out = self.net.BatchPermutation(
            [xform_shuffled, restore_bl], blob_out
        )

        return xform_out

    def RescaleAndDumplicateFeatureSingle(
        self,
        blobs_in,
        blob_out,
        blob_rois,
        src_spatial_scales,
        dst_spatial_scale
    ):
        """ Dumplicate feature maps for the refiner network.
        If use FPN, then rescale the different FPN level feature
        to a dst_spatial_scale. Then concancate the feature maps
        along the batch dimension

        Input blobs: res_...
        Input rois: mask_rois_fpn

        Output blobs: rois_global_feature
        """
        # Single scale feature
        src_sc = src_spatial_scales
        dst_sc = dst_spatial_scale
        blobs_in_list = [blobs_in, core.ScopedBlobReference(blob_rois)]
        name = 'RescaleAndDumplicateOp:' + ','.join(
            [str(b) for b in blobs_in_list]
        )

        blob_out = core.ScopedBlobReference(blob_out)

        xform_out = self.net.Python(
            RescaleAndDumplicateFeatureSingleOp(src_sc, dst_sc).forward,
            RescaleAndDumplicateFeatureSingleOp(src_sc, dst_sc).backward,
            grad_input_indices=[0]
        )(blobs_in_list, blob_out, name=name)

        return xform_out

    def RescaleAndDumplicateFeatureOld(
        self,
        blobs_in,
        blob_out,
        blob_rois,
        src_spatial_scales,
        dst_spatial_scale
    ):
        """ Dumplicate feature maps for the refiner network.
        If use FPN, then rescale the different FPN level feature
        to a dst_spatial_scale. Then concancate the feature maps
        along the batch dimension

        Input blobs: [fpn_<min>, ..., fpn_<max>]
        Input rois: [mask_rois_fpn<min>, ..., mask_rois_fpn<max>]

        Output blobs: rois_global_feature
        """
        # Add scoped blob

        if isinstance(blobs_in, list):
            # FPN cases: add RescaleAndDumplcateFeatureOp to each level
            # Since .net.Python can only use existing blob as input,
            # we create a blob to maintain some temporary parameters
            # and pass the blob to custom_op
            k_max = cfg.FPN.ROI_MAX_LEVEL
            k_min = cfg.FPN.ROI_MIN_LEVEL
            assert len(blobs_in) == k_max - k_min + 1
            dst_sc = dst_spatial_scale
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                src_sc = src_spatial_scales[k_max - lvl] # reversed order
                dst_sc = dst_spatial_scale

                bl_in = blobs_in[k_max - lvl] # came in reversed order
                bl_rois = core.ScopedBlobReference(blob_rois + '_fpn' + str(lvl))
                bl_in_list = [bl_in, bl_rois]
                name = 'RescaleAndDumplicateFeatureOp:' + ','.join(
                    [str(b) for b in bl_in_list]
                )

                bl_out = core.ScopedBlobReference(blob_out + '_fpn' + str(lvl))
                bl_out_list.append(bl_out)
                self.net.Python(
                    RescaleAndDumplicateFeatureOp(src_sc, dst_sc).forward
                )(bl_in_list, bl_out, name=name)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled, _ = self.net.Concat(
                bl_out_list, [blob_out + '_shuffled', '_concat_' + blob_out],
                axis=0
            )
            blob_rois = core.ScopedBlobReference(blob_rois)
            # Unshuffle to match rois from dataloader
            restore_bl = blob_rois + '_idx_restore_int32'
            xform_out = self.net.BatchPermutation(
                [xform_shuffled, restore_bl], blob_out
            )
        else:
            # Single scale feature
            src_sc = src_spatial_scales
            dst_sc = dst_spatial_scale
            blobs_in_list = [blobs_in, core.ScopedBlobReference(blob_rois)]
            name = 'RescaleAndDumplicateOp:' + ','.join(
                [str(b) for b in blobs_in_list]
            )

            blob_out = core.ScopedBlobReference(blob_out)

            xform_out = self.net.Python(
                RescaleAndDumplicateFeatureOp(src_sc, dst_sc).forward
            )(blobs_in_list, blob_out, name=name)

        # Only return the first blob (the transformed features)
        return xform_out

    def PoolingIndicatorFeatureSingle(
        self,
        blobs_in,
        blob_out,
        blob_rois,
        spatial_scale
    ):
        """ Pool indicator feature for the rois. Scale the roi with a
        factor and then create a feature map with size MxM.

        Input blobs: res_...
        Input rois: mask_rois_fpn

        Output blobs: rois_global_feature

        """
        M = cfg.REFINENET.RESOLUTION
        up_scale = cfg.REFINENET.UP_SCALE

        blobs_in_list = [blobs_in, core.ScopedBlobReference(blob_rois)]
        name = 'PoolingIndicatorFeatureSingleOp:' + ','.join(
            [str(b) for b in blobs_in_list]
        )

        blob_out = core.ScopedBlobReference(blob_out)

        xform_out = self.net.Python(
            PoolingIndicatorFeatureSingleOp(spatial_scale, up_scale, M).forward,
            PoolingIndicatorFeatureSingleOp(spatial_scale, up_scale, M).backward,
            grad_input_indices=[0]
        )(blobs_in_list, blob_out, name=name)

        return xform_out

    def PoolingIndicatorFeatureFPN(
        self,
        blobs_in,
        blob_out,
        blob_rois,
        spatial_scales
    ):
        """
        Pool indicator feature for the rois. Scale the roi with a
        factor and then create a feature map with size MxM.
        If use FPN, then call. Then concancate the feature maps
        along the batch dimension

        Input blobs: [fpn_<min>, ..., fpn_<max>]
        Input rois: [mask_rois_fpn<min>, ..., mask_rois_fpn<max>]

        Output blobs: rois_global_feature
        """
        M = cfg.REFINENET.RESOLUTION
        up_scale = cfg.REFINENET.UP_SCALE

        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        blob_fpn_rois = [
            core.ScopedBlobReference(blob_rois+'_fpn'+str(lvl))
            for lvl in range(k_min, k_max+1)
        ]

        scales = []
        blobs_in_list = []
        for lvl in range(k_min, k_max+1):
            blob_in = blobs_in[k_max - lvl] # reversed order
            scales.append(spatial_scales[k_max - lvl]) # reversed order
            blob_fpn_roi = blob_fpn_rois[lvl - k_min]
            blobs_in_list.append(blob_in)
            blobs_in_list.append(blob_fpn_roi)

        name = 'PoolingIndicatorFeatureFPNOp: ' + ','.join(
                [str(b) for b in blobs_in_list]
            )
        # ignore gradient for 'blob_rois'
        grad_input_indices = [2*(i-k_min) for i in range(k_min, k_max+1)]
        #grad_input_indices=[]

        blob_fpn_dumplicate_out = [
            core.ScopedBlobReference(blob_out+'_fpn'+str(lvl))
            for lvl in range(k_min, k_max+1)
        ]

        #Rescale and Dumplicate FPN feature
        blob_dumplicate_list = self.net.Python(
            PoolingIndicatorFeatureFPNOp(k_min,k_max,scales,up_scale,M).forward,
            PoolingIndicatorFeatureFPNOp(k_min,k_max,scales,up_scale,M).backward,
            grad_input_indices=grad_input_indices
        )(blobs_in_list, blob_fpn_dumplicate_out, name=name)

        # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
        xform_shuffled, _ = self.net.Concat(
            blob_dumplicate_list, [blob_out + '_shuffled', '_concat_' + blob_out],
            axis=0
        )
        # Unshuffle to match rois from dataloader
        restore_bl = core.ScopedBlobReference(blob_rois + '_idx_restore_int32')
        xform_out = self.net.BatchPermutation(
            [xform_shuffled, restore_bl], blob_out
        )

        return xform_out

    def RescaleFeatureMap(
        self,
        blobs_in,
        blob_out,
        dim_in,
        rescale_factor,
        spatial_scale=1. / 16.,
        sampling_ratio=0
    ):
        """ Rescale the feature map to a rescale_factor size.
        If use FPN, then rescale each FPN to a fixed size and
        concat them together.

        Else, pass the feature map.
        """

        method = 'RescaleFeatureMap'
        # get the output size
        dim_out = 0
        blob_data = core.ScopedBlobReference('data')

        if isinstance(blobs_in, list):
            # FPN case
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1

            bl_out_list = []
            for lvl in range(k_min, k_max+1):
                dim_out += dim_in
                bl_in = blobs_in[k_max - lvl] # reversed order
                sc = spatial_scale[k_max - lvl] # reversed order
                bl_rois = 'img_rois'
                bl_out = blob_out + '_fpn' + str(lvl)
                bl_out_list.append(bl_out)
                self.net.__getattr__(method)(
                    [bl_in, bl_rois, blob_data], [bl_out],
                    spatial_scale=sc,
                    rescale_factor=rescale_factor,
                    sampling_ratio=sampling_ratio
                )
            xform_out, _ = self.net.Concat(
                bl_out_list, [blob_out, '_concat_' + blob_out],
                axis=1
            )
        else:
            # Single feature level
            dim_out = dim_in
            # sampling_ratio is ignored for RoIPoolF
            xform_out = self.net.__getattr__(method)(
                [blobs_in, blob_rois], [blob_out],
                spatial_scale=spatial_scale,
                rescale_factor=rescale_factor,
                sampling_ratio=sampling_ratio
            )

        return xform_out, dim_out

    def GenerateAutoLearningIndicators(
        self,
        blobs_in,
        blob_out,
        blob_rois,
        up_scale,
        resolution
    ):
        """ Generate Indicators. Implemented in C++ and CUDA.
        The forward function is similar to GenerateLocalMaskIndicators.
        But This operator adds a backward function to allow e2e learning.
        The indicator here acts as an intermediate feature.
        blobs_in: mask_fcn_logits
        blob_out: mask_indicators

        op input: X, R, Data
        op output: Y
        """
        method = 'GenerateIndicators'

        blob_in_list = [blobs_in, blob_rois, 'data']
        blob_out = self.net.__getattr__(method)(
            blob_in_list, [blob_out],
            up_scale=float(up_scale),
            resolution=resolution
        )
        return blob_out

    def GenerateGlobalMaskIndicators(
        self,
        blobs_in,
        blob_out,
        blob_rois='mask_rois',
        dst_spatial_scale=1/16.
    ):
        """ Add mask indicators to the refine network. It maps the
        'mask_probs' into the input images' space, and narrow it down
        by the value 'scale'

        Input blobs: [data, mask_probs]
        Input rois: mask_rois
        Output blob: mask_indicators
        """
        blob_rois = core.ScopedBlobReference(blob_rois) # refer blob_rois
        blobs_in_list = blobs_in + [blob_rois]
        name = 'GenerateMaskIndicatorsOp:' + ','.join(
            [str(b) for b in blobs_in_list]
        )
        blob_out = core.ScopedBlobReference(blob_out)
        grad_input_indices=[0] # ignore gradient for blob_rois

        xform_out = self.net.Python(
            GenerateGlobalMaskIndicatorsOp(scale=dst_spatial_scale).forward,
            GenerateGlobalMaskIndicatorsOp(scale=dst_spatial_scale).backward,
            grad_input_indices=grad_input_indices
        )(blobs_in_list, blob_out, name=name)
        return xform_out


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio
