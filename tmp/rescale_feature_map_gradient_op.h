// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef RESCALE_FEATURE_MAP_OP_H_
#define RESCALE_FEATURE_MAP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class RescaleFeatureMapGradientOp final : public Operator<Context> {
 public:
  RescaleFeatureMapGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        rescale_factor_(OperatorBase::GetSingleArgument<int>("rescale_factor", 1)),
        sampling_ratio_(
            OperatorBase::GetSingleArgument<int>("sampling_ratio", -1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(rescale_factor_, 0);
    DCHECK_GE(sampling_ratio_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  float rescale_factor_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
};

} // namespace caffe2

#endif // RESCALE_FEATURE_MAP_OP_H_
