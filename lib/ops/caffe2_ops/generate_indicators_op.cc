#include "generate_indicators_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(GenerateIndicator, GenerateIndicatorsOp<float, CPUContext>);

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
    .Arg(
        "same_as_opencv",
        "(bool) default true; If set to true, produce same results as "
        "cv2.resize(). If set to false, then use coordinates similar to "
        "roi_align_op.cu. "
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
