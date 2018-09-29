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
#include "mask_iou_op.h"

namespace caffe2 {

namespace {
__global__ void ElementwiseMaxKernel(const int n, float* data, const float a) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    data[index] = (data[index] > a) ? data[index] : a;
  }
}

__global__ void PixelIoUKernel(
    const int n,
    const float* probs,
    const int* targets,
    float* intersections,
    float* unions) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1) {
      intersections[index] = 0.;
      unions[index] = 0.
    } else {
      intersections[index] = (probs[index] >= 0.5 && targets[index] == 1.)
      unions[index] = (probs[index] >= 0.5 || targets[index] == 1.)
    }
  }
}

} // namespace

template <>
bool MaskIoUOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto* IoUs = Output(0); 
  auto* MeanIoU = Output(1);

  CAFFE_ENFORCE(
      X.size() == T.size(),
      "probs and target must have the same size",
      "(",
      X.size(),
      " vs. ",
      T.size(),
      ")");
  
  int batch_size = X.dim32(0);
  vector<TIndex> out_shape(batch_size);

  pixels_inter_.ResizeLike(X);
  pixels_union_.ResizeLike(X);
  object_inter_.Resize(out_shape);
  object_union_.Resize(out_shape);
  
  IoUs->Resize(out_shape);
  MeanIoU->Resize(vector<TIndex>());

  // First calculate pixel IoUs
  PixelIoUKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      T.data<int>(),
      pixels_inter_.mutable_data<float>(),
      pixels_union_.mutable_data<float>());

  // Reduce pixel inter to object inter
  float* reduced_data = pixels_inter_.data<float>();
  float* collect_data = object_inter_.mutable_data<float>();
  int reduced_len = pixels_inter_.size() / batch_size;
  for (int i = 0; i < batch_size; i++) {
    math::Sum<float, CUDAContext>(
      reduced_len, reduced_data + i * reduced_len, collect_data, &context_);
    collect_data += 1;
  }

  // Reduce pixel union to object union
  reduced_data = pixels_union_.data<float>();
  collect_data = object_union_.mutable_data<float>();
  reduced_len = pixels_union_.size() / batch_size;
  for (int i = 0; i < batch_size; i++) {
    math::Sum<float, CUDAContext>(
      reduced_len, reduced_data + i * reduced_len, collect_data, &context_);
    collect_data += 1;
  }
  // and prevent division by zero if object_union are zero
  ElementwiseMaxKernel<<<
      CAFFE_GET_BLOCKS(object_union_.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      object_union_.size(), 
      object_union_.mutable_data<float>(), 
      1e-5);

  // Now we can calculate IoUs
  math::Div<float, CUDAContext>(
    IoUs->size(), object_inter_.data<float>(), object_union_.data<float>(), 
    IoUs->mutable_data<float>(), &context_);

  // And now calculate mean IoU
  math::Sum<float, CUDAContext>(
    IoUs->size(), IoUs->data<float>(), 
    MeanIoU->mutable_data<float>(), &context_);
  math::Scale<float, CUDAContext>(
    1, 1. / batch_size, MeanIoU->data<float>(), 
    MeanIoU->mutable_data<float>(), &context_);

  return true;
}

REGISTER_CUDA_OPERATOR(
    MaskIoU,
    MaskIoUOp<float, CUDAContext>);
} // namespace caffe2
