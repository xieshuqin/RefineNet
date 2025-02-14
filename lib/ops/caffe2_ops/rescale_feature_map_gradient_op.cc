#include "rescale_feature_map_gradient_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

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
void RescaleFeatureMapBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int /*num_rois*/,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois,
    int rois_cols) {
  DCHECK(rois_cols == 4 || rois_cols == 5);

  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * rois_cols;
    int roi_batch_ind = 0;
    if (rois_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

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

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          // atomic add is not needed for now since it is single threaded
          add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        } // if
      } // ix
    } // iy
  } // for
} // ROIAlignBackward

} // namespace

template <>
bool RescaleFeatureMapGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto& Data = Input(2); //Data
  auto& dY = Input(3); // Gradient of net w.r.t. output of "forward" op
                       // (aka "gradOutput")
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")

  int input_height_ = Data.dim32(2);
  int input_width_ = Data.dim32(3);
  pooled_height_ = int( rescale_factor_ * input_height_) ;
  pooled_width_ = int( rescale_factor_ * input_width_ );

  CAFFE_ENFORCE_EQ(R.ndim(), 2);
  // if R has 5 columns, the first column is the index, otherwise 0
  CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

  dX->ResizeLike(X);

  // Must zero-out dX before accumulating gradients
  // (TODO): Kaiming - is this safe?
  math::Set<float, CPUContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  if (dY.size() > 0) { // Handle possibly empty gradient if there were no rois
    RescaleFeatureMapBackwardFeature<float>(
        dY.size(),
        dY.data<float>(),
        R.dim32(0),
        spatial_scale_,
        X.dim32(1),
        X.dim32(2),
        X.dim32(3),
        pooled_height_,
        pooled_width_,
        sampling_ratio_,
        dX->mutable_data<float>(),
        R.data<float>(),
        R.dim32(1));
  }
  return true;
}

REGISTER_CPU_OPERATOR(RescaleFeatureMapGradient, RescaleFeatureMapGradientOp<float, CPUContext>);

// Input: X, rois, Data, dY (aka "gradOutput");
// Output: dX (aka "gradInput")
OPERATOR_SCHEMA(RescaleFeatureMapGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(0, "X", "See RoIPoolF.")
    .Input(1, "RoIs", "See RoIPoolF.")
    .Input(2, "Data", "Input data for the network. ")
    .Input(3, "dY", "Gradient of forward output 0 (Y)")
    .Output(0, "dX", "Gradient of forward input 0 (X)");

namespace {

class GetRescaleFeatureMapGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RescaleFeatureMapGradient",
        "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(RescaleFeatureMap, GetRescaleFeatureMapGradient);

} // namespace caffe2
