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
import cv2

from core.config import cfg
import utils.blob as blob_utils
import utils.boxes as box_utils
import utils.segms as segm_utils
import utils.keypoints as keypoint_utils

logger = logging.getLogger(__name__)

def add_refine_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx, data):
    """ Specify the different refined output type here. """
    refined_output_type = cfg.REFINENET.REFINE_OUTPUT_TYPE
    assert refined_output_type in {'Mask', 'Keypoint'}, \
        'Only Mask/Keypoint are supported to be refined'
    if refined_output_type == 'Mask':
        add_refine_mask_blobs(
            blobs, sampled_boxes, roidb, im_scale, batch_idx, data
        )
    else:
        add_refine_keypoints_blob(
            blobs, sampled_boxes, roidb, im_scale, batch_idx, data
        )


def add_refine_mask_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx, data):
    if cfg.REFINENET.LOCAL_MASK:
        add_refine_local_mask_blobs(
            blobs, sampled_boxes, roidb, im_scale, batch_idx, data
        )
    else:
        add_refine_global_mask_blobs(
            blobs, sampled_boxes, roidb, im_scale, batch_idx, data
        )


def add_refine_local_mask_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx, data):
    """Add RefineNet Mask specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    M = cfg.REFINENET.RESOLUTION
    up_scale = cfg.REFINENET.UP_SCALE
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    gt_classes = roidb['gt_classes'][polys_gt_inds]
    polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    boxes_from_polys = segm_utils.polys_to_boxes(polys_gt)
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1

    # Define size variables
    inp_h, inp_w = data.shape[2], data.shape[3]
    pad_img_h, pad_img_w = inp_h / im_scale, inp_w / im_scale

    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        masks = blob_utils.zeros((fg_inds.shape[0], M**2), int32=True)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # Expand the foreground rois by a factor of up_scale and
        # clip by the padded image boundary
        pad_rois_fg = box_utils.expand_boxes_by_scale(rois_fg, up_scale)
        pad_rois_fg = box_utils.clip_boxes_to_image(pad_rois_fg, pad_img_h, pad_img_w)

        if cfg.REFINENET.ONLY_USE_CROWDED_SAMPLES:
            # Only use crowded samples to train the RefineNet
            THRES = cfg.REFINENET.OVERLAP_THRESHOLD
            for i in range(rois_fg.shape[0]):
                overlap = overlaps_bbfg_bbpolys[i]
                if np.sum(overlap > THRES) > 1:
                    # if has multiple instances overlapped, use it for training
                    fg_polys_ind = fg_polys_inds[i]
                    poly_gt = polys_gt[fg_polys_ind]
                    pad_roi_fg = pad_rois_fg[i]
                    # Rasterize the portion of the polygon mask within the given fg roi
                    # to an M x M binary image
                    mask = segm_utils.polys_to_mask_wrt_box(poly_gt, pad_roi_fg, M)
                    mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
                    masks[i, :] = np.reshape(mask, M**2)

                else:   # Only one instance, then set label to be -1 (ignored)
                    masks[i, :] = -1
                    mask_class_labels[i] = 0
        elif cfg.REFINENET.ASSIGN_LARGER_WEIGHT_FOR_CROWDED_SAMPLES:
            loss_weights = blob_utils.ones((rois_fg.shape[0], ))
            for i in range(rois_fg.shape[0]):
                fg_polys_ind = fg_polys_inds[i]
                poly_gt = polys_gt[fg_polys_ind]
                pad_roi_fg = pad_rois_fg[i]
                class_label = mask_class_labels[i]

                # Rasterize the portion of the polygon mask within the given
                # fg roi to an M x M binary image
                mask = segm_utils.polys_to_mask_wrt_box(poly_gt, pad_roi_fg, M)
                mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
                masks[i, :] = np.reshape(mask, M**2)

                # And now determine the weight for each roi. If any instance
                # that is of the same class as the RoI, then we expect it to
                # be a hard sample and assigns a larger weight for this RoI
                for j in range(len(polys_gt)):
                    if j == fg_polys_ind:
                        continue
                    if gt_classes[j] == class_label: # only same class is valid
                        mask = segm_utils.polys_to_mask_wrt_box(
                            polys_gt[j], pad_roi_fg, M
                        )
                        # and check if has anypart fall inside the bbox
                        is_inside_bbox = (np.sum(mask) > 0)
                        if is_inside_bbox:
                            loss_weights[i] = cfg.REFINENET.WEIGHT_LOSS_CROWDED
                            break # early stop

        else:
            # add fg targets
            for i in range(rois_fg.shape[0]):
                fg_polys_ind = fg_polys_inds[i]
                poly_gt = polys_gt[fg_polys_ind]
                pad_roi_fg = pad_rois_fg[i]
                # Rasterize the portion of the polygon mask within the given fg roi
                # to an M x M binary image
                mask = segm_utils.polys_to_mask_wrt_box(poly_gt, pad_roi_fg, M)
                mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
                masks[i, :] = np.reshape(mask, M**2)

    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # pad_rois_fg is actually one background roi, but that's ok because ...
        pad_rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -blob_utils.ones((1, M**2), int32=True)
        # We label it with class = 0 (background)
        mask_class_labels = blob_utils.zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    pad_rois_fg = (pad_rois_fg.astype(np.float32))*im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((pad_rois_fg.shape[0], 1))
    pad_rois_fg = np.hstack((repeated_batch_idx, pad_rois_fg)).astype(np.int32)

    # Update blobs dict with Refine-Net blobs
    blobs['refined_mask_rois'] = pad_rois_fg
    blobs['roi_has_refined_mask_int32'] = roi_has_mask
    blobs['refined_masks_int32'] = masks

    if cfg.REFINENET.ASSIGN_LARGER_WEIGHT_FOR_CROWDED_SAMPLES:
        blobs['loss_weights'] = loss_weights


def add_refine_global_mask_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx, data):
    """Add RefineNet Mask specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    dst_scale = cfg.REFINENET.SPATIAL_SCALE
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    boxes_from_polys = segm_utils.polys_to_boxes(polys_gt)
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1

    # Define size variables
    inp_h, inp_w = data.shape[2], data.shape[3]
    out_h, out_w = int(inp_h * dst_scale), int(inp_w * dst_scale)

    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        masks = blob_utils.zeros((fg_inds.shape[0], out_h, out_w), int32=True)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # narrow scale and size
        scale = im_scale * dst_scale
        im_h, im_w = roidb['height'], roidb['width']
        im_label_h, im_label_w = int(im_h*scale), int(im_w*scale)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            poly_gt = polys_gt[fg_polys_ind]
            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an im_label_h x im_label_w binary image
            mask = segm_utils.polys_to_mask_scaled(poly_gt, im_h, im_w, scale)
            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            masks[i, 0:im_label_h, 0:im_label_w] = mask

        masks = np.reshape(masks, (-1,out_h*out_w))


    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -blob_utils.ones((1, out_h*out_w), int32=True)
        # We label it with class = 0 (background)
        mask_class_labels = blob_utils.zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Refine-Net blobs
    blobs['refined_mask_rois'] = rois_fg
    blobs['roi_has_refined_mask_int32'] = roi_has_mask
    blobs['refined_masks_int32'] = masks


def add_refine_keypoints_blobs(
        blobs, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx, data
    ):
    """Add Mask R-CNN keypoint specific blobs to the given blobs dictionary."""
    # Note: gt_inds must match how they're computed in
    # datasets.json_dataset._merge_proposal_boxes_into_roidb
    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    max_overlaps = roidb['max_overlaps']
    gt_keypoints = roidb['gt_keypoints']

    ind_kp = gt_inds[roidb['box_to_gt_ind_map']]
    within_box = _within_box(gt_keypoints[ind_kp, :, :], roidb['boxes'])
    vis_kp = gt_keypoints[ind_kp, 2, :] > 0
    is_visible = np.sum(np.logical_and(vis_kp, within_box), axis=1) > 0
    kp_fg_inds = np.where(
        np.logical_and(max_overlaps >= cfg.TRAIN.FG_THRESH, is_visible)
    )[0]

    kp_fg_rois_per_this_image = np.minimum(fg_rois_per_image, kp_fg_inds.size)
    if kp_fg_inds.size > kp_fg_rois_per_this_image:
        kp_fg_inds = np.random.choice(
            kp_fg_inds, size=kp_fg_rois_per_this_image, replace=False
        )

    sampled_fg_rois = roidb['boxes'][kp_fg_inds]
    box_to_gt_ind_map = roidb['box_to_gt_ind_map'][kp_fg_inds]

    # Define size variables
    up_scale = cfg.REFINENET.UP_SCALE
    inp_h, inp_w = data.shape[2], data.shape[3]
    pad_img_h, pad_img_w = inp_h / im_scale, inp_w / im_scale

    # Expand the foreground rois by a factor of up_scale and
    # clip by the padded image boundary
    pad_rois_fg = box_utils.expand_boxes_by_scale(sampled_fg_rois, up_scale)
    pad_rois_fg = box_utils.clip_boxes_to_image(pad_rois_fg, pad_img_h, pad_img_w)


    num_keypoints = gt_keypoints.shape[2]
    sampled_keypoints = -np.ones(
        (len(sampled_fg_rois), gt_keypoints.shape[1], num_keypoints),
        dtype=gt_keypoints.dtype
    )
    for ii in range(len(sampled_fg_rois)):
        ind = box_to_gt_ind_map[ii]
        if ind >= 0:
            sampled_keypoints[ii, :, :] = gt_keypoints[gt_inds[ind], :, :]
            assert np.sum(sampled_keypoints[ii, 2, :]) > 0

    heats, weights = keypoint_utils.keypoints_to_heatmap_labels(
        sampled_keypoints, pad_rois_fg, M=cfg.REFINENET.KRCNN.HEATMAP_SIZE
    )

    shape = (pad_rois_fg.shape[0] * cfg.KRCNN.NUM_KEYPOINTS, 1)
    heats = heats.reshape(shape)
    weights = weights.reshape(shape)

    pad_rois_fg = pad_rois_fg.astype(np.float32)
    pad_rois_fg *= im_scale
    pad_rois_fg.astype(np.int32)
    repeated_batch_idx = batch_idx * blob_utils.ones(
        (pad_rois_fg.shape[0], 1)
    )
    pad_rois_fg = np.hstack((repeated_batch_idx, pad_rois_fg))

    blobs['refined_keypoint_rois'] = pad_rois_fg
    blobs['refined_keypoint_locations_int32'] = heats.astype(np.int32, copy=False)
    blobs['refined_keypoint_weights'] = weights


def finalize_refined_keypoint_minibatch(blobs, valid):
    """Finalize the minibatch after blobs for all minibatch images have been
    collated.
    """
    min_count = cfg.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH
    num_visible_keypoints = np.sum(blobs['refined_keypoint_weights'])
    valid = (
        valid and len(blobs['refined_keypoint_weights']) > 0 and
        num_visible_keypoints > min_count
    )
    # Normalizer to use if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
    # See modeling.model_builder.add_keypoint_losses
    norm = num_visible_keypoints / (
        cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.BATCH_SIZE_PER_IM *
        cfg.TRAIN.FG_FRACTION * cfg.KRCNN.NUM_KEYPOINTS
    )
    blobs['refined_keypoint_loss_normalizer'] = np.array(norm, dtype=np.float32)
    return valid


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.

    points: Nx2xK
    boxes: Nx4
    output: NxK
    """
    x_within = np.logical_and(
        points[:, 0, :] >= np.expand_dims(boxes[:, 0], axis=1),
        points[:, 0, :] <= np.expand_dims(boxes[:, 2], axis=1)
    )
    y_within = np.logical_and(
        points[:, 1, :] >= np.expand_dims(boxes[:, 1], axis=1),
        points[:, 1, :] <= np.expand_dims(boxes[:, 3], axis=1)
    )
    return np.logical_and(x_within, y_within)



def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, out_h * out_w)
        to shape (#masks, #classes * out_h * out_w)
        to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    mask_size = masks.shape[1]

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -blob_utils.ones(
        (masks.shape[0], cfg.MODEL.NUM_CLASSES * mask_size), int32=True
    )

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = mask_size * cls
        end = start + mask_size
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets
