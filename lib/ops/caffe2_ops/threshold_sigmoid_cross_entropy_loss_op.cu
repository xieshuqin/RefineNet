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

#include "caffe2/core/context_gpu.h"
#include "threshold_sigmoid_cross_entropy_loss_op.h"

namespace caffe2 {

namespace {
__global__ void ElementwiseMaxKernel(const int n, float* data, const float a) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    data[index] = (data[index] > a) ? data[index] : a;
  }
}

__global__ void ThresholdSigmoidCrossEntropyLossKernel(
    const int n,
    const float logit_threshold,
    const float* logits,
    const int* targets,
    float* losses,
    float* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1 || logits[index] < logit_threshold) {
      losses[index] = 0.;
      counts[index] = 0.;
    } else {
      losses[index] =
          -1. * logits[index] * (targets[index] - (logits[index] >= 0)) +
          logf(
              1 +
              expf(logits[index] - 2 * logits[index] * (logits[index] >= 0)));
      counts[index] = 1.;
    }
  }
}

__global__ void ThresholdSigmoidCrossEntropyLossGradientKernel(
    const int n,
    const float logit_threshold,
    const float* logits,
    const int* targets,
    float* d_logits,
    float* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1 || logits[index] < logit_threshold) {
      d_logits[index] = 0.;
      counts[index] = 0.;
    } else {
      d_logits[index] = 1. / (1. + expf(-logits[index])) - targets[index];
      counts[index] = 1.;
    }
  }
}
} // namespace

template <>
bool ThresholdSigmoidCrossEntropyLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto* avg_loss = Output(0);

  const float logit_threshold = logf(threshold_ / (1. - threshold_));

  CAFFE_ENFORCE(
      X.size() == T.size(),
      "Logit and target must have the same size",
      "(",
      X.size(),
      " vs. ",
      T.size(),
      ")");
  avg_loss->Resize(vector<TIndex>());
  counts_.ResizeLike(X);
  losses_.ResizeLike(X);
  normalizer_.Resize(vector<TIndex>());
  ThresholdSigmoidCrossEntropyLossKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      logit_threshold, 
      X.data<float>(),
      T.data<int>(),
      losses_.mutable_data<float>(),
      counts_.mutable_data<float>());
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  if (normalize_) {
    float* normalizer_data = normalizer_.mutable_data<float>();
    math::Sum<float, CUDAContext>(
        counts_.size(), counts_.data<float>(), normalizer_data, &context_);
    // Prevent division by zero is all counts are zero
    ElementwiseMaxKernel<<<
        CAFFE_GET_BLOCKS(normalizer_.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(normalizer_.size(), normalizer_data, 1e-5);
    math::Div<float, CUDAContext>(
        1, avg_loss_data, normalizer_data, avg_loss_data, &context_);
  }
  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}

template <>
bool ThresholdSigmoidCrossEntropyLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& d_avg_loss = Input(2);
  auto* dX = Output(0);

  const float logit_threshold = logf(threshold_ / (1. - threshold_));

  dX->ResizeLike(X);
  counts_.ResizeLike(X);
  normalizer_.Resize(vector<TIndex>());
  ThresholdSigmoidCrossEntropyLossGradientKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      logit_threshold, 
      X.data<float>(),
      T.data<int>(),
      dX->mutable_data<float>(),
      counts_.mutable_data<float>());
  if (normalize_) {
    float* normalizer_data = normalizer_.mutable_data<float>();
    math::Sum<float, CUDAContext>(
        counts_.size(), counts_.data<float>(), normalizer_data, &context_);
    // Prevent division by zero is all counts are zero
    ElementwiseMaxKernel<<<
        CAFFE_GET_BLOCKS(normalizer_.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(normalizer_.size(), normalizer_data, 1e-5);
    math::Div<float, CUDAContext>(
        1,
        d_avg_loss.data<float>(),
        normalizer_data,
        normalizer_data,
        &context_);
    math::Scale<float, CUDAContext>(
        1, scale_, normalizer_data, normalizer_data, &context_);
    math::Scale<float, CUDAContext>(
        dX->size(),
        normalizer_data,
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
  } else {
    math::Scale<float, CUDAContext>(
        dX->size(),
        scale_,
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
    math::Scale<float, CUDAContext>(
        dX->size(),
        d_avg_loss.data<float>(),
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    ThresholdSigmoidCrossEntropyLoss,
    ThresholdSigmoidCrossEntropyLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ThresholdSigmoidCrossEntropyLossGradient,
    ThresholdSigmoidCrossEntropyLossGradientOp<float, CUDAContext>);
} // namespace caffe2
