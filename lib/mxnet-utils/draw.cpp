#include "draw.h"

#include <opencv2/imgproc.hpp>

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
    Scalar color = getRandomColor(res.cls);
    rectangle(img, res.bbox, color, 1);
    putText(img, to_string(res.cls) + ": " + to_string(res.score),
            res.bbox.tl() + Point(5, 20), 0, 0.3, color, 1, LINE_AA);
}

}  // namespace mxutils
