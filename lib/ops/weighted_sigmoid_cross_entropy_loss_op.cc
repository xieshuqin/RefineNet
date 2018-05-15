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

#include "weighted_sigmoid_cross_entropy_loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    WeightedSigmoidCrossEntropyLoss,
    WeightedSigmoidCrossEntropyLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    WeightedSigmoidCrossEntropyLossGradient,
    WeightedSigmoidCrossEntropyLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(WeightedSigmoidCrossEntropyLoss)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
A general version of Sigmoid Cross Entropy Loss but allows changing the weight 
between positive label and negative label. 
Compute sigmoid activations followed by averaged binary cross entropy loss for  
those whose activations are greater than threshold. The target values may be 
in {-1, 0, 1}, where -1 indicates that the corresponding sample should be 
ignored and {0, 1} correspond to the binary classes 0 and 1. By default the 
loss only considers the elements where the activations are above the 'threshold' 
op argument and then divided by the number of targets > -1 and then multiplied by
the `scale` op argument. The divisive normalization may be disable by setting
the op argument `normalize` to 0 (the multiplication by `scale` still takes
effect).

This op fuses sigmoid and cross entropy for numerical stability in both forward
and gradient computation.
)DOC")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Arg(
        "normalize",
        "(int) default 1; if true, divide the loss by the number of targets > "
        "-1.")
    .Arg(
        "pos_weight",
        "(float) default 1.0; weight for positive label "
        "for loss.")
    .Arg(
        "neg_weight",
        "(float) default 1.0; weight for negative label "
        "for loss.")
    .Input(
        0,
        "X",
        "Tensor of predicted logits (shape must be at least 1D).")
    .Input(
        1,
        "targets",
        "Tensor of targets of type int and same shape as logits X.")
    .Output(
        0,
        "loss",
        "Scalar loss.");

OPERATOR_SCHEMA(WeightedSigmoidCrossEntropyLossGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See WeightedSigmoidCrossEntropyLoss.")
    .Input(
        1,
        "targets",
        "See WeightedSigmoidCrossEntropyLoss.")
    .Input(
        2,
        "d_loss",
        "Gradient of forward output 0 (loss).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetWeightedSigmoidCrossEntropyLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "WeightedSigmoidCrossEntropyLossGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(
    WeightedSigmoidCrossEntropyLoss, 
    GetWeightedSigmoidCrossEntropyLossGradient
);

} // namespace caffe2
