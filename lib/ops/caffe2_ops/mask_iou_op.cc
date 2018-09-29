/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mask_iou_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    MaskIoU,
    MaskIoUOp<float, CPUContext>);

OPERATOR_SCHEMA(MaskIoU)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Compute Mask IoU value for the predicted masks and mask labels. Return the 
IoU value for each masks and the average IoU value for the entire batch.
)DOC")
    .Input(
        0,
        "X",
        "Tensor of predicted masks.")
    .Input(
        1,
        "targets",
        "Tensor of targets of type int and same shape as X.")
    .Output(
        0,
        "IoUs",
        "IoU value for each mask.")
    .Output(
        1,
        "MeanIoU",
        "Average IoU value for the entire batch"
        );

} // namespace caffe2
