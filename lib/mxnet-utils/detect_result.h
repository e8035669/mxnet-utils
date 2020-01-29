#ifndef _MXUTILS_DETECT_RESULT_H_
#define _MXUTILS_DETECT_RESULT_H_

#include <vector>

#include <opencv2/core.hpp>
#include <mxnet-cpp/MxNetCpp.h>

namespace mxutils {

struct DetectResult {
    int cls;
    float score;
    cv::Rect bbox;
};

std::vector<DetectResult> toDetectResult(const std::vector<mxnet::cpp::NDArray>& output, float thresh);


}  // namespace mxutils

#endif
