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

#include "mean_square_loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    MeanSquareLoss,
    MeanSquareLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    MeanSquareLossGradient,
    MeanSquareLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(MeanSquareLoss)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Mean Square Loss with weight. The scale argument multiples the output loss with 
a scale. 
)DOC")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Input(
        0,
        "X",
        "Tensor of predicted logits (shape must be at least 1D).")
    .Input(
        1,
        "targets",
        "Tensor of targets of type int and same shape as logits X.")
    .Input(
        2,
        "Weights",
        "Tensor of weights, automatically boardcast to the shape of X")
    .Output(
        0,
        "loss",
        "Scalar loss.");

OPERATOR_SCHEMA(MeanSquareLossGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See MeanSquareLoss.")
    .Input(
        1,
        "targets",
        "See MeanSquareLoss.")
    .Input(
        2,
        "Weights",
        "Tensor of weights, automatically boardcast to the shape of X")
    .Input(
        3,
        "d_loss",
        "Gradient of forward output 0 (loss).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetMeanSquareLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MeanSquareLossGradient",
        "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(MeanSquareLoss, GetMeanSquareLossGradient);

} // namespace caffe2
