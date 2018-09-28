// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef GENERATE_INDICATORS_OP_H_
#define GENERATE_INDICATORS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class GenerateIndicatorsOp final : public Operator<Context> {
 public:
  GenerateIndicatorsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        up_scale_(
            OperatorBase::GetSingleArgument<float>("up_scale", 1.)),
        resolution_(
            OperatorBase::GetSingleArgument<int>("resolution", 1.)) {
    DCHECK_GT(up_scale_, 0);
    DCHECK_GT(resolution_, 0);
    DCHECK(order_ == StorageOrder::NCHW);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  StorageOrder order_;
  float up_scale_;
  int resolution_;
};

} // namespace caffe2

#endif // GENERATE_INDICATORS_OP_H_
