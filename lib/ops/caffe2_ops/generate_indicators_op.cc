#include "generate_indicators_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

#ifdef CAFFE2_USE_MKL
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#endif // CAFFE2_USE_MKL

namespace caffe2 {
namespace {

template <typename T>
void expand_bbox_by_scale(
  const int n_rois,
  const T* bottom_rois,
  const int height,
  const int width,
  const float up_scale,
  T* top_rois,
  int roi_cols)  {
  // expand the bottom rois by up_scale 
  for (int n = 0; n < n_rois; ++n) {
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
    pad_x1 = std::min((T)width-1, std::max((T)0., pad_x1));
    pad_x2 = std::min((T)width-1, std::max((T)0., pad_x2));
    pad_y1 = std::min((T)height-1, std::max((T)0., pad_y1));
    pad_y2 = std::min((T)height-1, std::max((T)0., pad_y2));

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
void convert_coordinates(
  int n_rois,
  const T* bottom_rois,
  const T* top_rois,
  T* coordinates,
  int resolution,
  int roi_cols) {
  // convert the coordinates of bottom_rois to top_rois
  for (int n = 0; n < n_rois; ++n) {
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
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T bin_size_h,
    T bin_size_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {

      T x = pw * bin_size_w;
      T y = ph * bin_size_h;

      // deal with: inverse elements are out of feature map boundary
      if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        PreCalc<T> pc;
        pc.pos1 = 0;
        pc.pos2 = 0;
        pc.pos3 = 0;
        pc.pos4 = 0;
        pc.w1 = 0;
        pc.w2 = 0;
        pc.w3 = 0;
        pc.w4 = 0;
        pre_calc[pre_calc_index] = pc;
        pre_calc_index += 1;
        continue;
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
      T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

      // save weights and indeces
      PreCalc<T> pc;
      pc.pos1 = y_low * width + x_low;
      pc.pos2 = y_low * width + x_high;
      pc.pos3 = y_high * width + x_low;
      pc.pos4 = y_high * width + x_high;
      pc.w1 = w1;
      pc.w2 = w2;
      pc.w3 = w3;
      pc.w4 = w4;
      pre_calc[pre_calc_index] = pc;

      pre_calc_index += 1;
    }
  }
}

template <typename T>
void GenerateIndicatorsForward(
    const int nthreads,
    const T* bottom_data,
    const int channels,
    const int height,
    const int width,
    const int top_height,
    const int top_width,
    const int* coordinates,
    T* top_data,
    StorageOrder order) {
  // pooled_height, pooled_width is the size of indicator zone
  // top_height, top_width is the size of padded indicator output
  // height, width is the size of the original output.

  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  int n_rois = nthreads / channels / top_width / top_height;
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * top_width * top_height;

    // roi could have 4 or 5 columns
    const int* offset_coordinates = coordinates + n * 4; 
    int x1 = offset_coordinates[0];
    int y1 = offset_coordinates[1];
    int x2 = offset_coordinates[2];
    int y2 = offset_coordinates[3];

    int pooled_width = x2 - x1;
    int pooled_height = y2 - y1;

    T bin_size_h = static_cast<T>(height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(width) / static_cast<T>(pooled_width);

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        bin_size_h,
        bin_size_w,
        pre_calc);

    if (order == StorageOrder::NCHW) {
      for (int c = 0; c < channels; c++) {
        int index_n_c = index_n + c * top_width * top_height;
        const T* offset_bottom_data =
            bottom_data + (n * channels + c) * height * width;
        int pre_calc_index = 0;

        for (int ph = 0; ph < pooled_height; ph++) {
          for (int pw = 0; pw < pooled_width; pw++) {
            int index = index_n_c + (ph + y1) * top_width + (pw + x1);

            T output_val = 0.;
            PreCalc<T> pc = pre_calc[pre_calc_index];
            output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                    pc.w2 * offset_bottom_data[pc.pos2] +
                    pc.w3 * offset_bottom_data[pc.pos3] +
                    pc.w4 * offset_bottom_data[pc.pos4];

            pre_calc_index += 1;

            top_data[index] = output_val;

          } // for pw
        } // for ph
      } // for c
    } // if nchw

    if (order == StorageOrder::NHWC) {
      return ; // Not implement

    //  const T* offset_bottom_data =
    //      bottom_data + roi_batch_ind * channels * height * width;
    //  int pre_calc_index = 0;

    //  for (int ph = 0; ph < pooled_height; ph++) {
    //    for (int pw = 0; pw < pooled_width; pw++) {
    //      EVecXf output_vals = EVecXf::Zero(channels);

    //      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
    //        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
    //          PreCalc<T> pc = pre_calc[pre_calc_index];

    //          ConstEigenVectorMap<T> data_1(
    //              offset_bottom_data + channels * pc.pos1, channels);
    //          ConstEigenVectorMap<T> data_2(
    //              offset_bottom_data + channels * pc.pos2, channels);
    //          ConstEigenVectorMap<T> data_3(
    //              offset_bottom_data + channels * pc.pos3, channels);
    //          ConstEigenVectorMap<T> data_4(
    //              offset_bottom_data + channels * pc.pos4, channels);

    //          output_vals += pc.w1 * data_1 + pc.w2 * data_2 + pc.w3 * data_3 +
    //              pc.w4 * data_4;

    //          pre_calc_index += 1;
    //        }
    //      }
    //      output_vals /= count;

    //      int index_nhw = index_n + (ph * pooled_width + pw) * channels;
    //      std::memcpy(
    //          top_data + index_nhw, output_vals.data(), channels * sizeof(T));
    //    } // for pw
    //  } // for ph
    } // if nhwc

  } // for n
}

} // namespace

template <>
bool GenerateIndicatorsOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Input data for generating indicators
  auto& R = Input(1); // RoIs
  auto& Data = Input(2); // Data
  auto* Y = Output(0); // Indicators

  int input_height_ = Data.dim32(2);
  int input_width_ = Data.dim32(3);

  if (R.size() == 0) {
    // Handle empty rois
    if (order_ == StorageOrder::NCHW) {
      Y->Resize(0, X.dim32(1), resolution_, resolution_);
    } else if (order_ == StorageOrder::NHWC) {
      Y->Resize(0, resolution_, resolution_, X.dim32(3));
    }
    // The following mutable_data calls are needed to allocate the tensors
    Y->mutable_data<float>();
    return true;
  }

  CAFFE_ENFORCE_EQ(R.ndim(), 2);
  // if R has 5 columns, the first column is the index, otherwise 0
  CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

  // padded the RoIs by the up_scale factor
  TensorCPU pad_R(R.dims());
  expand_bbox_by_scale<float>(
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
  TensorCPU coordinates(dims);

  convert_coordinates<float>(
    R.dim32(0),
    R.data<float>(),
    pad_R.data<float>(),
    coordinates.mutable_data<float>(),
    resolution_,
    R.dim32(1));

  // Perform the resize and copying operation
  if (order_ == StorageOrder::NCHW) {
    Y->Resize(R.dim32(0), X.dim32(1), resolution_, resolution_);
    int output_size = Y->size();
    GenerateIndicatorsForward<float>(
        output_size,
        X.data<float>(),
        X.dim32(1),
        X.dim32(2),
        X.dim32(3),
        resolution_,
        resolution_,
        coordinates.data<int>(),
        Y->mutable_data<float>(),
        order_);
  } else if (order_ == StorageOrder::NHWC) {
    return false; // Not implement error

    //Y->Resize(R.dim32(0), resolution_, resolution_, X.dim32(3));
    //int output_size = Y->size();
    //GenerateIndicatorsForward<float>(
    //    output_size,
    //    X.data<float>(),
    //    spatial_scale_,
    //    X.dim32(3),
    //    X.dim32(1),
    //    X.dim32(2),
    //    pooled_height,
    //    pooled_width,
    //    R.data<float>(),
    //    R.dim32(1),
    //    Y->mutable_data<float>(),
    //    order_);
  }

  return true;
}

REGISTER_CPU_OPERATOR(GenerateIndicator, GenerateIndicatorsOp<float, CPUContext>);

#ifdef CAFFE2_HAS_MKL_DNN
REGISTER_MKL_OPERATOR(
    GenerateIndicators,
    mkl::MKLFallbackOp<GenerateIndicatorsOp<float, CPUContext>>);
#endif // CAFFE2_HAS_MKL_DNN

// Input: data, mask_probs, mask_rois; Output: mask_indicators
OPERATOR_SCHEMA(GenerateIndicators)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Generate Indicators operation as used in RefineNet.
)DOC")
    .Arg(
        "up_scale",
        "(float) default 1.0; Up scale factor for padding the rois "
        )
    .Arg(
        "resolution",
        "(int) default 1.0; resolution for the indicators"
        )
    .Input(
        0, 
        "X", 
        "4D input data of shape (N, C, H, W). ")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Input(
        2, 
        "Data", 
        "Input image data of shape (N, 3, H, W).")
    .Output(
        0,
        "Y",
        "4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element "
        "is a pooled feature map cooresponding to the r-th RoI.");

} // namespace caffe2
