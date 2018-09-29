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

#include "sigmoid_accuracy_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SigmoidAccuracy,
    SigmoidAccuracyOp<float, CPUContext>);

OPERATOR_SCHEMA(SigmoidAccuracy)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Compute accuracy for the predicted sigmoid probability and 
labels.
)DOC")
    .Input(
        0,
        "X",
        "Tensor of predicted probs (shape must be at least 1D).")
    .Input(
        1,
        "targets",
        "Tensor of targets of type int and same shape as X.")
    .Output(
        0,
        "accuracy",
        "Accuracy.");

} // namespace caffe2
