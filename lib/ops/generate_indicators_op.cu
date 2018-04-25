#include "generate_indicators_op.h"

#include <stdio.h>
#include <cfloat>
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void expand_bbox_by_scale(
  const int nthreads,
  const T* bottom_rois,
  const int height,
  const int width,
  const float up_scale,
  T* top_rois,
  int roi_cols)  {
  // expand the bottom rois by up_scale 
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T x1 = offset_bottom_rois[0];
    T y1 = offset_bottom_rois[1];
    T x2 = offset_bottom_rois[2];
    T y2 = offset_bottom_rois[3];

    T roi_width = x2 - x1;
    T roi_height = y2 - y1;
    T center_x = (x1 + x2) / 2;
    T center_y = (y1 + y2) / 2;

    // expand the size by up_scale factor 
    T pad_roi_width = roi_width * up_scale;
    T pad_roi_height = roi_height * up_scale;
    T pad_x1 = center_x - pad_roi_width / 2;
    T pad_y1 = center_y - pad_roi_height / 2;
    T pad_x2 = center_x + pad_roi_width / 2;
    T pad_y2 = center_y + pad_roi_height / 2;

    // clip to image boundary
    pad_x1 = min((T)(width-1), max((T)0., pad_x1));
    pad_x2 = min((T)(width-1), max((T)0., pad_x2));
    pad_y1 = min((T)(height-1), max((T)0., pad_y1));
    pad_y2 = min((T)(height-1), max((T)0., pad_y2));

    // write to top_rois
    T* offset_top_rois = top_rois + n * roi_cols;
    if (roi_cols == 5) {
      offset_top_rois[0] = roi_batch_ind;
      offset_top_rois++;
    }

    offset_top_rois[0] = pad_x1;
    offset_top_rois[1] = pad_y1;
    offset_top_rois[2] = pad_x2;
    offset_top_rois[3] = pad_y2;
  }
}

template <typename T>
__global__ void convert_coordinates(
  const int nthreads,
  const T* bottom_rois,
  const T* top_rois,
  T* coordinates,
  int resolution,
  int roi_cols) {
  // convert the coordinates of bottom_rois to top_rois
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    T x1 = offset_bottom_rois[0]; 
    T y1 = offset_bottom_rois[1];
    T x2 = offset_bottom_rois[2];
    T y2 = offset_bottom_rois[3];

    const T* offset_top_rois = top_rois + n * roi_cols;
    if (roi_cols == 5) { offset_top_rois++; }
    T pad_x1 = offset_top_rois[0];
    T pad_y1 = offset_top_rois[1];
    T pad_x2 = offset_top_rois[2];
    T pad_y2 = offset_top_rois[3];

    T pad_width = pad_x2 - pad_x1 + 1;
    T pad_height = pad_y2 - pad_y1 + 1;

    T converted_x1 = (x1 - pad_x1) / pad_width * resolution;
    T converted_x2 = (x2 - pad_x1) / pad_width * resolution;
    T converted_y1 = (y1 - pad_y1) / pad_height * resolution;
    T converted_y2 = (y2 - pad_y1) / pad_height * resolution;

    T* offset_coordinates = coordinates + n * 4;
    offset_coordinates[0] = converted_x1;
    offset_coordinates[1] = converted_y1;
    offset_coordinates[2] = converted_x2;
    offset_coordinates[3] = converted_y2;
  }

}

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}


template <typename T>
__global__ void GenerateIndicatorsForward(
    const int nthreads,
    const T* bottom_data,
    const int channels,
    const int height,
    const int width,
    const int top_height,
    const int top_width,
    const int* coordinates,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % top_width;
    int ph = (index / top_width) % top_height;
    int c = (index / top_width / top_height) % channels;
    int n = index / top_width / top_height / channels;

    const int* offset_coordinates = coordinates + n * 4;
    int x1 = offset_coordinates[0];
    int y1 = offset_coordinates[1];
    int x2 = offset_coordinates[2];
    int y2 = offset_coordinates[3];

    // zero if outside the coordinate zone
    if (pw < x1 || pw > x2 || ph < y1 || ph > y2) {
        top_data[index] = 0;
    }
    else {
        int pooled_height = y2 - y1 + 1;
        int pooled_width = x2 - x1 + 1;
        T bin_size_h = static_cast<T>(height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(width) / static_cast<T>(pooled_width);

        const T* offset_bottom_data = 
            bottom_data + (n * channels + c) * height * width;

        const T x = (pw - x1) * bin_size_w;
        const T y = (ph - y1) * bin_size_h;
        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        top_data[index] = val;
    }
  }
}

} // namespace

template <>
bool GenerateIndicatorsOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data for generating indicators
  auto& R = Input(1); // RoIs
  auto& Data = Input(2); // Data
  auto* Y = Output(0); // Indicators

  int input_height_ = Data.dim32(2);
  int input_width_ = Data.dim32(3);

  if (R.size() == 0) {
    // Handle empty rois
    Y->Resize(0, X.dim32(1), resolution_, resolution_);
    // The following mutable_data calls are needed to allocate the tensors
    Y->mutable_data<float>();
    return true;
  }

  int n_rois = R.dim32(0)
  // padded the RoIs by the up_scale factor
  TensorCUDA pad_R(R.dims());
  expand_bbox_by_scale<float>
      <<<CAFFE_GET_BLOCKS(n_rois),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          R.dim32(0),
          R.data<float>(),
          input_height_,
          input_width_,
          up_scale_,
          pad_R.mutable_data<float>(),
          R.dim32(1));

  // convert the coordinates of R to pad_R in resolution_
  std::vector<int> dims(2);
  dims[0] = R.dim32(0); dims[1] = 4;
  TensorCUDA coordinates(dims);
  convert_coordinates<float>
      <<<CAFFE_GET_BLOCKS(n_rois),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          R.dim32(0),
          R.data<float>(),
          pad_R.data<float>(),
          coordinates.mutable_data<float>(),
          resolution_,
          R.dim32(1));

  Y->Resize(R.dim32(0), X.dim32(1), resolution_, resolution_);
  int output_size = Y->size();
  GenerateIndicatorsForward<float>
      <<<CAFFE_GET_BLOCKS(output_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          output_size,
          X.data<float>(),
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          resolution_,
          resolution_,
          coordinates.data<int>(),
          Y->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(GenerateIndicators, 
  GenerateIndicatorsOp<float, CUDAContext>);
} // namespace caffe2
