from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, utils
from hypothesis import assume, given, settings
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import cv2


def expand_boxes_by_scale(xyxy, scale):
    """ Scale xyxy boxes by a scale """
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        ctr_x, ctr_y = xyxy[0] + 0.5*w, xyxy[1] + 0.5*h
        x1, x2 = ctr_x - w*scale/2, ctr_x + w*scale/2
        y1, y2 = ctr_y - h*scale/2, ctr_y + h*scale/2
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        size = xyxy[:, 2:4] - xyxy[:, 0:2]
        ctr = xyxy[:, 0:2] + 0.5*size
        return np.hstack((ctr-size*scale/2, ctr+size*scale/2))
    else:
        raise TypeError('Argument xyxy must be a list, typle or numpy array.')


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(width, np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height, np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def convert_coordinate(box_from, box_to, M):
    """ Convert the coordinate of box_from into the
    coordinate axis of box_to.
    The box_from and box_to are in the same coordinate axis.
    """
    box_from = box_from.astype(np.float32)
    box_to = box_to.astype(np.float32)

    box_to_ul = box_to[:, 0:2]
    box_to_size = box_to[:, 2:4] - box_to[:, 0:2]

    box_from_ul = box_from[:, 0:2]
    box_from_br = box_from[:, 2:4]

    converted_ul_norm = (box_from_ul - box_to_ul) / box_to_size
    converted_br_norm = (box_from_br - box_to_ul) / box_to_size

    convert_coord_norm = np.hstack((converted_ul_norm, converted_br_norm))
    convert_coord = (convert_coord_norm * M)
    convert_coord = np.rint(convert_coord).astype(np.int32)

    return convert_coord


def generate_indicators_ref(*inputs):
    """ inputs are blobs_in and args
        blobs_in: [data, mask_probs, mask_rois]
        args: {'resolution': M, 'up_scale': up_scale}
    """
    assert inputs
    args = inputs[-1]
    inputs = inputs[:-1]

    mask_probs = inputs[0]
    mask_rois = inputs[1]
    data = inputs[2]

    M = args['resolution']
    up_scale = args['up_scale']
    num_cls = mask_probs.shape[1]
    num_rois = mask_rois.shape[0]
    mask_indicators = np.zeros((num_rois, M, M, num_cls), dtype='float32')

    # preparing data
    height, width = data.shape[2], data.shape[3]
    mask_probs_NHWC = mask_probs.transpose((0,2,3,1))
    rois = mask_rois[:, 1:5] # ignore batch_id
    pad_rois = expand_boxes_by_scale(rois, up_scale)
    pad_rois = clip_boxes_to_image(pad_rois, height, width)
    # calculate converted coordinates
    converted_coords = convert_coordinate(rois, pad_rois, M)
    for i in range(num_rois):
        mask_prob = mask_probs_NHWC[i]
        coords = converted_coords[i]
        shape = (coords[2]-coords[0], coords[3]-coords[1]) # w,h
        mask_prob_resize = cv2.resize(mask_prob, shape)
        if mask_prob_resize.shape[2] == 1:
            mask_prob_resize = mask_prob_resize[:, :, np.newaxis]
        mask_indicators[i, coords[1]:coords[3], coords[0]:coords[2]] = \
            mask_prob_resize

    swap_order = (0, 3, 1, 2)
    mask_indicators = mask_indicators.transpose(swap_order)
    #print('mask_indicators:', mask_indicators[1][1])

    outputs = [mask_indicators]
    return outputs


class TestGenerateIndicatorsOp(hu.HypothesisTestCase):

    @given(proposal_count=st.integers(min_value=2, max_value=16),
           roi_canonical_scale=st.integers(min_value=100, max_value=300),
           up_scale=st.floats(1,6),
           resolution=st.integers(min_value=28, max_value=28),
           **hu.gcs)
    def test_generate_indicators(
        self,
        proposal_count,
        roi_canonical_scale,
        up_scale,
        resolution,
        gc, dc):

        assume(gc.device_type == caffe2_pb2.CUDA)
        # np.random.seed(0)

        Data = np.zeros(shape=(2,3,800,1000), dtype=np.float32)
        X = np.random.rand(proposal_count, 4, 28, 28).astype(np.float32)
        #print('X: ', X[1][1])
        roi = (
            roi_canonical_scale *
            np.random.rand(proposal_count, 5).astype(np.float32)
        )
        for i in range(proposal_count):
            # Make RoIs have positive area, since they
            # are in the format [[batch_idx, x0, y0, x1, y2], ...]
            roi[i][3] += roi[i][1]
            roi[i][4] += roi[i][2]
        roi = clip_boxes_to_image(roi, Data.shape[2], Data.shape[3])

        op = core.CreateOperator(
            "GenerateIndicators",
            ["X", "R", "Data"],
            ["Y"],
            #arg=[
            #    utils.MakeArgument("up_scale", up_scale),
            #    utils.MakeArgument("resolution", resolution),
            #],
            up_scale=up_scale,
            resolution=resolution,
            device_option=gc
        )
        args = {
            'up_scale' : up_scale,
            'resolution' : resolution
        }

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, roi, Data] + [args],
            reference=generate_indicators_ref,
        )

        # Check over multiple devices
        if (gc.device_type == caffe2_pb2.CUDA):
             self.assertGradientChecks(gc, op, [X, roi, Data], 0, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
