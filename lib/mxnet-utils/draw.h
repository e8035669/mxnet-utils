#ifndef _MXUTILS_DRAW_H_
#define _MXUTILS_DRAW_H_

#include <opencv2/core.hpp>

#include "detect_result.h"
#include "opencv2/core/mat.hpp"

namespace mxutils {

cv::Scalar getRandomColor(int id);

void drawBbox(cv::InputOutputArray img, const DetectResult& res);

void drawBbox(cv::InputOutputArray img, const DetectResult& res, const std::vector<std::string>& class_names);

}

#endif
