#include "generate_indicators_gradient_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

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
void bilinear_interpolate_gradient(
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
    const int /*index*/ /* index for debug only*/) {
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

template <class T>
inline void add(const T& val, T* address) {
  *address += val;
}

template <typename T>
void GenerateIndicatorsBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int channels,
    const int height,
    const int width,
    const int top_height,
    const int top_width,
    const int* coordinates,
    T* bottom_diff) {

  for (int index = 0; index < nthreads; index++) {
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

    if (pw >= x1 && ph >= y1 && pw <= x2 && pw <= y2) {

      int pooled_width = x2 - x1 + 1;
      int pooled_height = y2 - y1 + 1;

      T bin_size_h = static_cast<T>(height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(width) / static_cast<T>(pooled_width);

      T* offset_bottom_diff =
          bottom_diff + (n * channels + c) * height * width;

      int top_offset = (n * channels + c) * top_height * top_width;
      const T* offset_top_diff = top_diff + top_offset;
      const T top_diff_this_bin = offset_top_diff[ph * top_width + pw];

      const T y = (ph - y1) * bin_size_h;
      const T x = (pw - x1) * bin_size_w;

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

      T g1 = top_diff_this_bin * w1 ;
      T g2 = top_diff_this_bin * w2 ;
      T g3 = top_diff_this_bin * w3 ;
      T g4 = top_diff_this_bin * w4 ;

      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        // atomic add is not needed for now since it is single threaded
        add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
        add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
        add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
        add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
      } // if
    } // if 
  } // for
} // ROIAlignBackward

} // namespace

template <>
bool GenerateIndicatorsGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto& Data = Input(2); //Data
  auto& dY = Input(3); // Gradient of net w.r.t. output of "forward" op
                       // (aka "gradOutput")
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")

  int input_height_ = Data.dim32(2);
  int input_width_ = Data.dim32(3);

  CAFFE_ENFORCE_EQ(R.ndim(), 2);
  // if R has 5 columns, the first column is the index, otherwise 0
  CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

  // Padded the RoIs by the up_scale factor
  TensorCPU pad_R(R.dims());
  expand_bbox_by_scale<float>(
    R.dim32(0),
    R.data<float>(),
    input_height_,
    input_width_,
    up_scale_,
    pad_R.mutable_data<float>(),
    R.dim32(1));

  // Convert the coordinates of R to pad_R in resolution_
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

  dX->ResizeLike(X);

  // Must zero-out dX before accumulating gradients
  // (TODO): Kaiming - is this safe?
  math::Set<float, CPUContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  if (dY.size() > 0) { // Handle possibly empty gradient if there were no rois
    GenerateIndicatorsBackwardFeature<float>(
        dY.size(),
        dY.data<float>(),
        X.dim32(1),
        X.dim32(2),
        X.dim32(3),
        resolution_,
        resolution_,
        coordinates.data<int>(),
        dX->mutable_data<float>());
  }
  return true;
}

REGISTER_CPU_OPERATOR(
  GenerateIndicatorsGradient, GenerateIndicatorsGradientOp<float, CPUContext>);

// Input: X, rois, Data, dY (aka "gradOutput");
// Output: dX (aka "gradInput")
OPERATOR_SCHEMA(GenerateIndicatorsGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(0, "X", "See RoIPoolF.")
    .Input(1, "RoIs", "See RoIPoolF.")
    .Input(2, "Data", "Input data for the network. ")
    .Input(3, "dY", "Gradient of forward output 0 (Y)")
    .Output(0, "dX", "Gradient of forward input 0 (X)");

namespace {

class GetGenerateIndicatorsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GenerateIndicatorsGradient",
        "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(GenerateIndicators, GetGenerateIndicatorsGradient);

} // namespace caffe2
