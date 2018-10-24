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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import gradient_checker
from caffe2.python import workspace

import utils.c2
import utils.logging


class MeanSquareLossTest(unittest.TestCase):
    def test_forward_and_gradient(self):
        X = np.random.randn((512, 17, 56, 56)).astype(np.float32)
        Y = np.random.randn((512, 17, 56, 56)).astype(np.float32)
        Weights = np.random.randn((512, 17, 1)).astype(np.float32)
        scale = np.random.random()

        op = core.CreateOperator(
            'MeanSquareLoss', ['X', 'Y', 'Weights'],
            ['loss'],
            scale=scale
        )

        gc = gradient_checker.GradientChecker(
            stepsize=0.005,
            threshold=0.005,
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0)
        )

        res, grad, grad_estimated = gc.CheckSimple(
            op, [X, Y, Weights, scale], 0, [0]
        )

        self.assertTrue(
            grad.shape == grad_estimated.shape,
            'Fail check: grad.shape != grad_estimated.shape'
        )

        # To inspect the gradient and estimated gradient:
        # np.set_printoptions(precision=3, suppress=True)
        # print('grad:')
        # print(grad)
        # print('grad_estimated:')
        # print(grad_estimated)

        self.assertTrue(res)


if __name__ == '__main__':
    utils.c2.import_detectron_ops()
    assert 'MeanSquareLoss' in workspace.RegisteredOperators()
    utils.logging.setup_logging(__name__)
    unittest.main()
