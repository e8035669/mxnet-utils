#ifndef _MXUTILS_DETECT_RESULT_H_
#define _MXUTILS_DETECT_RESULT_H_

#include <vector>

#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/core.hpp>

#include "predictor.h"

namespace mxutils {

struct DetectResult {
    int cls;
    float score;
    cv::Rect bbox;
};

std::vector<DetectResult> toDetectResult(
    const std::vector<mxnet::cpp::NDArray>& output, float thresh);

class DetectPredictor : public Predictor {
   public:
    DetectPredictor(const mxnet::cpp::Context& ctx) : Predictor(ctx) {}

    std::vector<DetectResult> predict(cv::InputArray input, float thresh);
};

}  // namespace mxutils

#endif
