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
#include "sigmoid_accuracy_op.h"

namespace caffe2 {

namespace {
__global__ void ElementwiseMaxKernel(const int n, float* data, const float a) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    data[index] = (data[index] > a) ? data[index] : a;
  }
}

__global__ void SigmoidAccuracyKernel(
    const int n,
    const float* probs,
    const int* targets,
    float* true_positives,
    float* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1) {
      true_positives[index] = 0.;
      counts[index] = 0.;
    } else {
      true_positives[index] = (targets[index] == 1. && probs[index] >= 0.5) + 
          (targets[index] == 0 && probs[index] < 0.5);
      counts[index] = 1.;
    }
  }
}
} // namespace

template <>
bool SigmoidAccuracyOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto* accuracy = Output(0);

  CAFFE_ENFORCE(
      X.size() == T.size(),
      "Logit and target must have the same size",
      "(",
      X.size(),
      " vs. ",
      T.size(),
      ")");
  // Resize 
  true_positives_.ResizeLike(X);
  counts_.ResizeLike(X);
  match_.Resize(vector<TIndex>());
  total_.Resize(vector<TIndex>());

  accuracy->Resize(vector<TIndex>());

  SigmoidAccuracyKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      T.data<int>(),
      true_positives_.mutable_data<float>(),
      counts_.mutable_data<float>());

  // Sum true_positives to match
  float* match_data = match_.mutable_data<float>();
  math::Sum<float, CUDAContext>(
      true_positives_.size(), true_positives_.data<float>(), 
      match_data, &context_);

  // Sum counts to total
  float* total_data = total_.mutable_data<float>();
  math::Sum<float, CUDAContext>(
      counts_.size(), counts_.data<float>(), total_data, &context_);
  // Prevent division by zero is all counts are zero
  ElementwiseMaxKernel<<<
      CAFFE_GET_BLOCKS(total_.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(total_.size(), total_data, 1e-5);

  // Get accuracy
  math::Div<float, CUDAContext>(
      1, match_data, total_data, 
      accuracy->mutable_data<float>(), &context_);

  return true;
}

REGISTER_CUDA_OPERATOR(
    SigmoidAccuracy,
    SigmoidAccuracyOp<float, CUDAContext>);
} // namespace caffe2
