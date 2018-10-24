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
#include "mean_square_loss_op.h"

namespace caffe2 {

namespace {
__global__ void ElementwiseMaxKernel(const int n, float* data, const float a) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    data[index] = (data[index] > a) ? data[index] : a;
  }
}

__global__ void MeanSquareLossKernel(
    const int n,
    const float* logits,
    const float* targets,
    float* losses) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    losses[index] = 0.5 * (logits[index] - targets[index]) * 
      (logits[index] - targets[index]); 
  }
}

__global__ void MeanSquareLossGradientKernel(
    const int n,
    const float* logits,
    const float* targets,
    float* d_logits) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    d_logits[index] = logits[index] - targets[index];
  }
}

__global__ void StripedScaleKernel(
    const int n,
    const int D,  
    const float* x,
    const float* alpha, 
    float* y) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int k = index / D;
    y[index] = x[index] * alpha[k];
 }
}
} // namespace

template <>
bool MeanSquareLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& Weights = Input(2);
  auto* avg_loss = Output(0);
  int D = X.size() / Weights.size();

  CAFFE_ENFORCE(
      X.size() == T.size(),
      "Logit and target must have the same size",
      "(",
      X.size(),
      " vs. ",
      T.size(),
      ")");
  avg_loss->Resize(vector<TIndex>());
  losses_.ResizeLike(X);

  MeanSquareLossKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      T.data<float>(),
      losses_.mutable_data<float>());

  StripedScaleKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      losses_.size(),
      D,
      losses_.data<float>(),
      Weights.data<float>(),
      losses_.mutable_data<float>());


  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}

template <>
bool MeanSquareLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& Weights = Input(2);
  auto& d_avg_loss = Input(3);
  auto* dX = Output(0);
  int D = X.size() / Weights.size(); 

  // Compute difference
  dX->ResizeLike(X);
  MeanSquareLossGradientKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      T.data<float>(),
      dX->mutable_data<float>());

  // Multiply by weight
  StripedScaleKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      dX->size(),
      D,
      dX->data<float>(),
      Weights.data<float>(),
      dX->mutable_data<float>());

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

  return true;
}

REGISTER_CUDA_OPERATOR(
    MeanSquareLoss,
    MeanSquareLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    MeanSquareLossGradient,
    MeanSquareLossGradientOp<float, CUDAContext>);
} // namespace caffe2
