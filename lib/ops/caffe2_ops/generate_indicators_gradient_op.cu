#include "generate_indicators_gradient_op.h"

#include <stdio.h>
#include <cfloat>
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void expand_boxes_and_clip_boundary(
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

    T roi_width_half = (x2 - x1) / 2;
    T roi_height_half = (y2 - y1) / 2;
    T center_x = (x1 + x2) / 2;
    T center_y = (y1 + y2) / 2;

    // expand the size by up_scale factor 
    T pad_roi_width_half = roi_width_half * up_scale;
    T pad_roi_height_half = roi_height_half * up_scale;
    T pad_x1 = center_x - pad_roi_width_half;
    T pad_y1 = center_y - pad_roi_height_half;
    T pad_x2 = center_x + pad_roi_width_half;
    T pad_y2 = center_y + pad_roi_height_half;

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

    T converted_x1 = int((x1 - pad_x1) / pad_width * resolution);
    T converted_x2 = int((x2 - pad_x1) / pad_width * resolution);
    T converted_y1 = int((y1 - pad_y1) / pad_height * resolution);
    T converted_y2 = int((y2 - pad_y1) / pad_height * resolution);

    T* offset_coordinates = coordinates + n * 4;
    offset_coordinates[0] = converted_x1;
    offset_coordinates[1] = converted_y1;
    offset_coordinates[2] = converted_x2;
    offset_coordinates[3] = converted_y2;
  }

}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

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

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void GenerateIndicatorsBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int channels,
    const int height,
    const int width,
    const int top_height,
    const int top_width,
    const T* coordinates,
    T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % top_width;
    int ph = (index / top_width) % top_height;
    int c = (index / top_width / top_height) % channels;
    int n = index / top_width / top_height / channels;

    const T* offset_coordinates = coordinates + n * 4;
    int x1 = int(offset_coordinates[0]);
    int y1 = int(offset_coordinates[1]);
    int x2 = int(offset_coordinates[2]);
    int y2 = int(offset_coordinates[3]);

    if (pw >= x1 && pw <= x2 && ph >= y1 && ph <= y2) {
      int pooled_width = x2 - x1 + 1;
      int pooled_height = y2 - y1 + 1;
      T bin_size_h = static_cast<T>(height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(width) / static_cast<T>(pooled_width);

      T* offset_bottom_diff = 
          bottom_diff + (n * channels + c) * height * width;

      int top_offset = (n * channels + c) * top_height * top_width;
      const T* offset_top_diff = top_diff + top_offset;
      const T top_diff_this_bin = offset_top_diff[ph * top_width + pw];

      //const T x = (pw - x1) * bin_size_w;
      //const T y = (ph - y1) * bin_size_h;
      const T y = (ph - y1 + 0.5) * bin_size_h - 0.5; // some magic trick
      const T x = (pw - x1 + 0.5) * bin_size_w - 0.5;

      T w1, w2, w3, w4;
      int x_low, x_high, y_low, y_high;
      bilinear_interpolate_gradient(
          height,
          width,
          y,
          x,
          w1,
          w2,
          w3,
          w4,
          x_low,
          x_high,
          y_low,
          y_high,
          index);

      T g1 = top_diff_this_bin * w1;
      T g2 = top_diff_this_bin * w2;
      T g3 = top_diff_this_bin * w3;
      T g4 = top_diff_this_bin * w4;

      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        gpu_atomic_add(
            static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
        gpu_atomic_add(
            static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
        gpu_atomic_add(
            static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
        gpu_atomic_add(
            static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
      }
    }
  } // CUDA_1D_KERNEL_LOOP
} // GenerateIndicatorBackward

} // namespace

template <>
bool GenerateIndicatorsGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto& Data = Input(2); //Data
  auto& dY = Input(3); // Gradient of net w.r.t. output of "forward" op
                       // (aka "gradOutput")
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")

  int input_height_ = Data.dim32(2);
  int input_width_ = Data.dim32(3);

  int n_rois = R.dim32(0);
  // padded the RoIs by the up_scale factor
  TensorCUDA pad_R(R.dims());
  expand_boxes_and_clip_boundary<float>
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

  dX->ResizeLike(X);

  // Must zero-out dX before accumulating gradients
  // (TODO): Kaiming - is this safe?
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  if (dY.size() > 0) { // Handle possibly empty gradient if there were no rois
    GenerateIndicatorsBackwardFeature<float>
        <<<CAFFE_GET_BLOCKS(dY.size()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            dY.size(),
            dY.data<float>(),
            X.dim32(1),
            X.dim32(2),
            X.dim32(3),
            resolution_,
            resolution_,
            coordinates.data<float>(),
            dX->mutable_data<float>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    GenerateIndicatorsGradient,
    GenerateIndicatorsGradientOp<float, CUDAContext>);
} // namespace caffe2
