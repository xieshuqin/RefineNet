//@author: xuchao
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

#ifndef THRESHOLD_SIGMOID_HINGLE_LOSS_OP_H_
#define THRESHOLD_SIGMOID_HINGLE_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class ThresholdSigmoidHingleLossOp final : public Operator<Context> {
 public:
  ThresholdSigmoidHingleLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        normalize_(OperatorBase::GetSingleArgument<int>("normalize", 1)),
        high_threshold_(OperatorBase::GetSingleArgument<float>("high_threshold", 0.99)),
        low_threshold_(OperatorBase::GetSingleArgument<float>("low_threshold", 0.01)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE(normalize_ == 0 || normalize_ == 1);
    CAFFE_ENFORCE(low_threshold_ > 0 && low_threshold_ < 1);
    CAFFE_ENFORCE(high_threshold_ > 0 && high_threshold_ < 1);
    CAFFE_ENFORCE(high_threshold_ > low_threshold_);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int normalize_;
  float low_threshold_;
  float high_threshold_;
  Tensor<Context> losses_;
  Tensor<Context> counts_;
  Tensor<Context> normalizer_;
};

template <typename T, class Context>
class ThresholdSigmoidHingleLossGradientOp final : public Operator<Context> {
 public:
  ThresholdSigmoidHingleLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        normalize_(OperatorBase::GetSingleArgument<int>("normalize", 1)),
        high_threshold_(OperatorBase::GetSingleArgument<float>("high_threshold", 0.99)),
        low_threshold_(OperatorBase::GetSingleArgument<float>("low_threshold", 0.01)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE(normalize_ == 0 || normalize_ == 1);
    CAFFE_ENFORCE(low_threshold_ > 0 && low_threshold_ < 1);
    CAFFE_ENFORCE(high_threshold_ > 0 && high_threshold_ < 1);
    CAFFE_ENFORCE(high_threshold_ > low_threshold_);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int normalize_;
  float low_threshold_;
  float high_threshold_;
  Tensor<Context> counts_;
  Tensor<Context> normalizer_;
};

} // namespace caffe2

#endif // THRESHOLD_SIGMOID_HINGLE_LOSS_OP_H_
