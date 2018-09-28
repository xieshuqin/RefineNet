// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef GENERATE_INDICATORS_OP_H_
#define GENERATE_INDICATORS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class GenerateIndicatorsGradientOp final : public Operator<Context> {
 public:
  GenerateIndicatorsGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        up_scale_(
            OperatorBase::GetSingleArgument<float>("up_scale", 1.)),
        resolution_(
          OperatorBase::GetSingleArgument<int>("resolution", 1)) {
    DCHECK_GT(up_scale_, 0);
    DCHECK_GT(resolution_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float up_scale_;
  int resolution_;
};

} // namespace caffe2

#endif // RESCALE_FEATURE_MAP_OP_H_
