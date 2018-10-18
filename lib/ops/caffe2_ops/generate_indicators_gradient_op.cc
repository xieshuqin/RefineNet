#include "generate_indicators_gradient_op.h"

namespace caffe2 {

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
