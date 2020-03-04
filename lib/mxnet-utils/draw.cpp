#include "draw.h"

#include <opencv2/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

namespace mxutils {

cv::Scalar getRandomColor(int id) {
    float hue = id * 0.618033988749895;
    hue = fmod(hue, 1.0);
    Mat hsvMat(1, 1, CV_32FC3, Scalar(hue * 255, 0.75, 0.95));
    Mat bgrMat;
    cvtColor(hsvMat, bgrMat, COLOR_HSV2BGR);
    Scalar color = bgrMat.at<Vec3f>(0) * 255;
    return color;
}

void drawBbox(cv::InputOutputArray img, const DetectResult& res) {
    drawBbox(img, res, {});
}

void drawBbox(cv::InputOutputArray img, const DetectResult& res,
              const std::vector<std::string>& class_names) {
    Scalar color = getRandomColor(res.cls);
    rectangle(img, res.bbox, color, 1);
    string text = to_string(res.cls) + ": " + to_string(res.score);
    if (res.cls < (int)class_names.size()) {
        text = class_names[(size_t)res.cls] + ": " + to_string(res.score);
    }

    putText(img, text, res.bbox.tl() + Point(5, 10), 0, 0.3, color, 1, LINE_AA);
}

}  // namespace mxutils
