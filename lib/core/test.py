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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Inference functionality for most Detectron models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import logging
import numpy as np

from caffe2.python import core
from caffe2.python import workspace
import pycocotools.mask as mask_util

from core.config import cfg
from utils.timer import Timer
import modeling.FPN as fpn
import utils.blob as blob_utils
import utils.boxes as box_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils

logger = logging.getLogger(__name__)


def im_detect_all(model, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scales = im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scales = im_detect_bbox(model, im, box_proposals)
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes)
        else:
            masks = im_detect_mask(model, im_scales, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1]
        )
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect_keypoints_aug(model, im, boxes)
        else:
            heatmaps = im_detect_keypoints(model, im_scales, boxes)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    if cfg.MODEL.PRN_ON and boxes.shape[0] > 0:
        timers['im_detect_prn'].tic()
        prn_probs = im_detect_prn(model, im_scales, boxes)
        timers['im_detect_prn'].toc()

        timers['misc_prn'].tic()
        roi_needs_refine = prn_results(cls_boxes, prn_probs, boxes)
        timers['misc_prn'].toc()
    else:
        roi_needs_refine = None

    if cfg.MODEL.REFINE_MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_refined_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            refined_masks = im_detect_refined_mask_aug(model, im, boxes)
        else:
            refined_masks = im_detect_refined_mask(model, im_scales, boxes)
        timers['im_detect_refined_mask'].toc()

        timers['misc_refined_mask'].tic()
        cls_refined_segms = refined_segm_results(
            cls_boxes, refined_masks, boxes, im_scales, im.shape[0], im.shape[1]
        )
        timers['misc_refined_mask'].toc()
    else:
        cls_refined_segms = None

    if cfg.MODEL.REFINE_MASK_ON and cfg.MODEL.PRN_ON and boxes.shape[0] > 0:
        # And another condition
        if cfg.TEST.USE_PRN_FOR_REFINE:
            # Merge cls_refined_segms with cls_segms
            cls_refined_segms = merge_refined_results_with_normal_results(
                cls_boxes, cls_segms, cls_refined_segms, roi_needs_refine
            )

    if cfg.MODEL.REFINE_KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_refined_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            refined_heatmaps = im_detect_refined_keypoints_aug(model, im, boxes)
        else:
            refined_heatmaps = im_detect_refined_keypoints(model, im_scales, boxes)
        timers['im_detect_refined_keypoints'].toc()

        timers['misc_refined_keypoints'].tic()
        cls_refined_keyps = refined_keypoint_results(
            cls_boxes, refined_heatmaps, boxes, im_scales
        )
        timers['misc_refined_keypoints'].toc()
    else:
        cls_refined_keyps = None

    return cls_boxes, cls_segms, cls_keyps, cls_refined_segms, cls_refined_keyps


def im_conv_body_only(model, im):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale_factors = _get_image_blob(im)
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale_factors


def im_detect_bbox(model, im, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.net.Proto().name)

    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        assert len(im_scales) == 1, \
            'Only single-image / single-scale batch implemented'
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS
        )
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scales


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _im_scales_hf = im_detect_bbox_hflip(
            model, im, box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scales_i = im_detect_bbox(model, im, box_proposals)
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scales_i


def im_detect_bbox_hflip(model, im, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scales = im_detect_bbox(
        model, im_hf, box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scales


def im_detect_bbox_scale(
    model, im, scale, max_size, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, box_proposals
        )
    else:
        scores_scl, boxes_scl, _ = im_detect_bbox(model, im, box_proposals)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
    model, im, aspect_ratio, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model, im_ar, box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(model, im_ar, box_proposals_ar)

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_mask(model, im_scales, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scales)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')
    ).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_mask_aug(model, im, boxes):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    im_scales_i = im_conv_body_only(model, im)
    masks_i = im_detect_mask(model, im_scales_i, boxes)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(model, im, boxes)
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return masks_c


def im_detect_mask_hflip(model, im, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    masks_hf = im_detect_mask(model, im_scales, boxes_hf)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes masks at the given scale."""

    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform mask detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        masks_scl = im_detect_mask_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        masks_scl = im_detect_mask(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        masks_ar = im_detect_mask(model, im_scales, boxes_ar)

    return masks_ar


def im_detect_keypoints(model, im_scales, boxes):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scales)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.keypoint_net.Proto().name)

    pred_heatmaps = workspace.FetchBlob(core.ScopedName('kps_score')).squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_aug(model, im, boxes):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """

    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    im_scales = im_conv_body_only(model, im)
    heatmaps_i = im_detect_keypoints(model, im_scales, boxes)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(model, im, boxes)
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALES[0]
        us_scl = scale > cfg.TEST.SCALES[0]
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c


def im_detect_keypoints_hflip(model, im, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    heatmaps_hf = im_detect_keypoints(model, im_scales, boxes_hf)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""

    # Store the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        heatmaps_scl = im_detect_keypoints(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False
):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        heatmaps_ar = im_detect_keypoints(model, im_scales, boxes_ar)

    return heatmaps_ar


def im_detect_refined_mask(model, im_scales, boxes):
    """ Head function for local/global mask indicator detection"""
    if cfg.REFINENET.LOCAL_MASK:
        return im_detect_refined_local_mask(model, im_scales, boxes)
    else:
        return im_detect_refined_global_mask(model, im_scales, boxes)


def im_detect_refined_local_mask(model, im_scales, boxes):
    """Infer refined instance segmentation masks. This function must be called
    after **im_detect_mask** as it assumes that the Caffe2 workspace is already
    populated with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox

    Returns:
        pred_refined_masks (ndarray): R x K x M x M array of class specific
            soft masks, where M is the refined mask resolution, defined in
            cfg.REFINENET.RESOLUTION
            The output must be processed by the function refined_segm_results
            to convert into hard masks in the original image coordinate space)

    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.REFINENET.RESOLUTION
    num_cls = cfg.MODEL.NUM_CLASSES

    if boxes.shape[0] == 0:
        pred_refined_masks = np.zeros((0, M, M), np.float32)
        return pred_refined_masks

    workspace.RunNet(model.refine_mask_net.Proto().name)

    # Fetch masks
    pred_refined_masks = workspace.FetchBlob(
        core.ScopedName('refined_mask_probs')
    ).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_refined_masks = pred_refined_masks.reshape([-1, num_cls, M, M])
    else:
        pred_refined_masks = pred_refined_masks.reshape([-1, 1, M, M])

    return pred_refined_masks

def im_detect_refined_global_mask(model, im_scales, boxes):
    """Infer refined instance segmentation masks. This function must be called
    after **im_detect_mask** as it assumes that the Caffe2 workspace is already
    populated with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox

    Returns:
        pred_refined_masks (ndarray): R x K x sH x sW array of class specific
            soft masks, where s is the down-sampling scale, defined in
            cfg.REFINENET.SPATIAL_SCALE and H, W is the size of the 'data'
            blob.
            The output must be processed by the function refined_segm_results
            to convert into hard masks in the original image coordinate space)

    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    down_scale = cfg.REFINENET.SPATIAL_SCALE
    num_cls = cfg.MODEL.NUM_CLASSES
    data = workspace.FetchBlob(core.ScopedName('data'))
    inp_h, inp_w = data.shape[2], data.shape[3]
    out_h, out_w = int(inp_h*down_scale), int(inp_w*down_scale)

    if boxes.shape[0] == 0:
        pred_refined_masks = np.zeros((0, out_h, out_w), np.float32)
        return pred_refined_masks

    workspace.RunNet(model.refine_mask_net.Proto().name)

    # Fetch masks
    pred_refined_masks = workspace.FetchBlob(
        core.ScopedName('refined_mask_probs')
    ).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_refined_masks = pred_refined_masks.reshape([-1, num_cls, out_h, out_w])
    else:
        pred_refined_masks = pred_refined_masks.reshape([-1, 1, out_h, out_w])

    return pred_refined_masks


def im_detect_refined_mask_aug(model, im, boxes):
    """Performs refined mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    refined_masks_ts = []

    # Compute masks for the original image (identity transform)
    im_scales_i = im_conv_body_only(model, im)
    masks_i = im_detect_mask(model, im_scales_i, boxes)
    refined_masks_i = im_detect_refined_mask(model, im_scales_i, boxes)
    refined_masks_ts.append(refined_masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        refined_masks_hf = im_detect_refined_mask_hflip(model, im, boxes)
        refined_masks_ts.append(refined_masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        refined_masks_scl = im_detect_refined_mask_scale(
            model, im, scale, max_size, boxes
        )
        refined_masks_ts.append(refined_masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            refined_masks_scl_hf = im_detect_refined_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            refined_masks_ts.append(refined_masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        refined_masks_ar = im_detect_refined_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        refined_masks_ts.append(refined_masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            refined_masks_ar_hf = im_detect_refined_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            refined_masks_ts.append(refined_masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        refined_masks_c = np.mean(refined_masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        refined_masks_c = np.amax(refined_masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in refined_masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        refined_masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return refined_masks_c


def im_detect_refined_mask_hflip(model, im, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    masks_hf = im_detect_mask(model, im_scales, boxes_hf)
    refined_masks_hf = im_detect_refined_mask(model, im_scales, boxes_hf)

    # Invert the predicted soft masks
    refined_masks_inv = refined_masks_hf[:, :, :, ::-1]

    return refined_masks_inv


def im_detect_refined_mask_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes refined masks at the given scale."""

    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform mask detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        refined_masks_scl = im_detect_refined_mask_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        masks_scl = im_detect_mask(model, im_scales, boxes)
        refined_masks_scl = im_detect_refined_mask(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return refined_masks_scl


def im_detect_refined_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        refined_masks_ar = im_detect_refined_mask_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        masks_ar = im_detect_mask(model, im_scales, boxes_ar)
        refined_masks_ar = im_detect_refined_mask(model, im_scales, boxes_ar)

    return refined_masks_ar


def im_detect_refined_keypoints(model, im_scales, boxes):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.REFINENET.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    workspace.RunNet(model.refine_keypoint_net.Proto().name)

    pred_heatmaps = workspace.FetchBlob(
        core.ScopedName('refined_kps_score')
    ).squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_refined_keypoints_aug(model, im, boxes):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """

    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    im_scales = im_conv_body_only(model, im)
    im_detect_keypoints(model, im_scales, boxes)
    heatmaps_i = im_detect_refined_keypoints(model, im_scales, boxes)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_refined_keypoints_hflip(model, im, boxes)
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALES[0]
        us_scl = scale > cfg.TEST.SCALES[0]
        heatmaps_scl = im_detect_refined_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_refined_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_refined_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_refined_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        # There might be a bug here. The boxes are not expanded so it
        # may not be correct.
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c


def im_detect_refined_keypoints_hflip(model, im, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_refined_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    im_detect_keypoints(model, im_scales, boxes_hf)
    heatmaps_hf = im_detect_refined_keypoints(model, im_scales, boxes_hf)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_refined_keypoints_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""

    # Store the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        heatmaps_scl = im_detect_refined_keypoints_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        im_detect_keypoints(model, im_scales, boxes)
        heatmaps_scl = im_detect_refined_keypoints(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return heatmaps_scl


def im_detect_refined_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False
):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_refined_keypoints_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        im_detect_keypoints(model, im_scales, boxes_ar)
        heatmaps_ar = im_detect_refined_keypoints(model, im_scales, boxes_ar)

    return heatmaps_ar


def im_detect_prn(model, im_scales, boxes):
    """Infer refined instance segmentation masks. This function must be called
    after **im_detect_mask** as it assumes that the Caffe2 workspace is already
    populated with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox

    Returns:
        pred_refined_masks (ndarray): R x K x M x M array of class specific
            soft masks, where M is the refined mask resolution, defined in
            cfg.REFINENET.RESOLUTION
            The output must be processed by the function refined_segm_results
            to convert into hard masks in the original image coordinate space)

    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    num_cls = cfg.MODEL.NUM_CLASSES

    if boxes.shape[0] == 0:
        prn_probs = np.zeros((0, ), np.float32)
        return prn_probs

    workspace.RunNet(model.prn_net.Proto().name)

    # Fetch prn_probs
    prn_probs = workspace.FetchBlob(core.ScopedName('prn_probs')).squeeze()

    # And feed a dummy roi_needs_refine to the workspace
    workspace.FeedBlob(
        core.ScopedName('roi_needs_refine_int32'),
        np.ones((boxes.shape[0], ), dtype=np.int32)
    )

    if cfg.PRN.CLS_SPECIFIC_LABEL:
        prn_probs = prn_probs.reshape([-1, num_cls])
    else:
        prn_probs = prn_probs.reshape([-1, 1])

    return prn_probs


def combine_heatmaps_size_dep(hms_ts, ds_ts, us_ts, boxes, heur_f):
    """Combines heatmaps while taking object sizes into account."""
    assert len(hms_ts) == len(ds_ts) and len(ds_ts) == len(us_ts), \
        'All sets of hms must be tagged with downscaling and upscaling flags'

    # Classify objects into small+medium and large based on their box areas
    areas = box_utils.boxes_area(boxes)
    sm_objs = areas < cfg.TEST.KPS_AUG.AREA_TH
    l_objs = areas >= cfg.TEST.KPS_AUG.AREA_TH

    # Combine heatmaps computed under different transformations for each object
    hms_c = np.zeros_like(hms_ts[0])

    for i in range(hms_c.shape[0]):
        hms_to_combine = []
        for hms_t, ds_t, us_t in zip(hms_ts, ds_ts, us_ts):
            # Discard downscaling predictions for small and medium objects
            if sm_objs[i] and ds_t:
                continue
            # Discard upscaling predictions for large objects
            if l_objs[i] and us_t:
                continue
            hms_to_combine.append(hms_t[i])
        hms_c[i] = heur_f(hms_to_combine)

    return hms_c


def box_results_with_nms_and_limit(scores, boxes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def refined_segm_results(cls_boxes, refined_masks, ref_boxes, im_scales, im_h, im_w):
    # header function for local/global indicator
    if cfg.REFINENET.LOCAL_MASK:
        return refined_local_segm_results(cls_boxes, refined_masks, ref_boxes, im_scales, im_h, im_w)
    else:
        return refined_global_segm_results(cls_boxes, refined_masks, im_scales, im_h, im_w)


def refined_local_segm_results(cls_boxes, masks, ref_boxes, im_scales, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0

    # Since the refined mask is done on the padded image, we need to
    # copy the output to the padded image. Therefore, we need to
    # get the padded image size.
    data = workspace.FetchBlob(core.ScopedName('data'))
    pad_h, pad_w = data.shape[2], data.shape[3]
    pad_img_h, pad_img_w = int(pad_h / im_scales), int(pad_w / im_scales)

    # The ref_boxes are with regard to mask_rois, we need to scale it
    # up to get the boxes for indicator, then clip it to the size of
    # padded image
    up_scale = cfg.REFINENET.UP_SCALE
    ref_boxes = box_utils.expand_boxes_by_scale(ref_boxes, up_scale)
    ref_boxes = box_utils.clip_boxes_to_image(ref_boxes, pad_img_h, pad_img_w)

    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.REFINENET.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def refined_global_segm_results(cls_boxes, refined_masks, im_scales, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_refined_segms = [[] for _ in range(num_classes)]
    mask_ind = 0

    refined_scale = cfg.REFINENET.SPATIAL_SCALE
    scale = im_scales * refined_scale
    inv_scale = 1. / scale

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        refined_segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                mask = refined_masks[mask_ind, j]
            else:
                mask = refined_masks[mask_ind, 0]

            mask = cv2.resize(mask, None, None, fx=inv_scale, fy=inv_scale)
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = mask[0:im_h, 0:im_w]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            refined_segms.append(rle)

            mask_ind += 1

        cls_refined_segms[j] = refined_segms

    assert mask_ind == refined_masks.shape[0]
    return cls_refined_segms


def merge_refined_results_with_normal_results(
        cls_boxes, normal_results, refined_results, roi_needs_refine
    ):
    """ A post-processing function to merge the normal results
    with refined_results if we have a binary roi_needs_refine to
    tell whether the roi needs refinement. If roi_needs_refine is 0,
    then use the normal results, else use the refined_results.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_merged_results = [[] for _ in range(num_classes)]
    roi_ind = 0

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        merged_results = []
        for i in range(cls_boxes[j].shape[0]):
            if roi_needs_refine[roi_ind] == 0:
                # use normal results
                merged_results.append(normal_results[j][i])
            else:
                # use refined results
                merged_results.append(refined_results[j][i])

            roi_ind += 1

        cls_merged_results[j] = merged_results

    assert roi_ind == roi_needs_refine.shape[0]
    return cls_merged_results


def prn_results(cls_boxes, prn_probs, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    roi_needs_refine = np.zeros((ref_boxes.shape[0], ))
    prn_ind = 0

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.PRN.CLS_SPECIFIC_LABEL:
                prn_prob = prn_probs[prn_ind, j]
            else:
                prn_prob = prn_probs[prn_ind, 0]

            roi_needs_refine[prn_ind] = 0 if prn_prob < 0.5 else 1
            prn_ind += 1

    assert prn_ind == ref_boxes.shape[0]
    assert prn_ind == prn_probs.shape[0]
    return roi_needs_refine


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def refined_keypoint_results(cls_boxes, pred_heatmaps, ref_boxes, im_scales):

    # Since the refined mask is done on the padded image, we need to
    # copy the output to the padded image. Therefore, we need to
    # get the padded image size.
    data = workspace.FetchBlob(core.ScopedName('data'))
    pad_h, pad_w = data.shape[2], data.shape[3]
    pad_img_h, pad_img_w = int(pad_h / im_scales), int(pad_w / im_scales)

    # The ref_boxes are with regard to mask_rois, we need to scale it
    # up to get the boxes for indicator, then clip it to the size of
    # padded image
    up_scale = cfg.REFINENET.UP_SCALE
    ref_boxes = box_utils.expand_boxes_by_scale(ref_boxes, up_scale)
    ref_boxes = box_utils.clip_boxes_to_image(ref_boxes, pad_img_h, pad_img_w)
    ref_boxes = ref_boxes.astype(np.float32)

    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (ndarray): array of image scales (relative to im) used
            in the image pyramid
    """
    processed_ims, im_scale_factors = blob_utils.prep_im_for_blob(
        im, cfg.PIXEL_MEANS, cfg.TEST.SCALES, cfg.TEST.MAX_SIZE
    )
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :]**2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if cfg.MODEL.FASTER_RCNN and rois is None:
        height, width = blobs['data'].shape[2], blobs['data'].shape[3]
        scale = im_scale_factors[0]
        blobs['im_info'] = np.array([[height, width, scale]], dtype=np.float32)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors
