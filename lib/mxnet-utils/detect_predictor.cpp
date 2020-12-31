#include "detect_predictor.h"

#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace mxnet::cpp;

namespace mxutils {

std::vector<DetectResult> toDetectResult(
    const std::vector<mxnet::cpp::NDArray>& output, float thresh) {
    if (output.size() != 3) throw runtime_error("need 3 outputs.");
    for (size_t i = 0; i < 3; i++) {
        if (output[i].GetContext().GetDeviceType() != mxnet::cpp::kCPU)
            throw runtime_error("not in cpu");
        output[i].WaitToRead();
    }

    NDArray ids = output[0];
    NDArray scores = output[1];
    NDArray bboxes = output[2];

    vector<DetectResult> results;

    size_t num = bboxes.GetShape()[1];
    for (size_t i = 0; i < num; ++i) {
        float score = scores.At(0, i, 0);
        float cls = ids.At(0, i, 0);
        if (score > thresh) {
            DetectResult res;
            res.cls = static_cast<int>(cls);
            res.score = score;
            Point tl(bboxes.At(0, i, 0), bboxes.At(0, i, 1));
            Point br(bboxes.At(0, i, 2), bboxes.At(0, i, 3));
            res.bbox = Rect(tl, br);
            results.push_back(res);
        }
    }

    return results;
}

std::vector<DetectResult> DetectPredictor::predict(cv::InputArray input,
                                                   float thresh) {
    Shape inputShape = this->getInputSize();
    Mat inputMat = input.getMat();
    if (inputMat.channels() != (int)inputShape[3]) {
        throw runtime_error("Channel not match");
    }
    double fx = 1.0, fy = 1.0;
    if (inputMat.size() != Size(inputShape[2], inputShape[1])) {
        fx = (double)inputMat.cols / inputShape[2];
        fy = (double)inputMat.rows / inputShape[1];
        cv::resize(inputMat, inputMat, Size(inputShape[2], inputShape[1]));
    }
    Mat floatMat;
    inputMat.convertTo(floatMat, CV_32F);

    NDArray inputnd(inputShape, Context::cpu());
    inputnd.SyncCopyFromCPU((float*)floatMat.data, inputShape.Size());

    vector<NDArray> outputs = Predictor::predict(inputnd);

    vector<DetectResult> results = toDetectResult(outputs, thresh);
    for (DetectResult& r : results) {
        Point tl = r.bbox.tl();
        Point br = r.bbox.br();
        tl.x *= fx;
        tl.y *= fy;
        br.x *= fx;
        br.y *= fy;
        r.bbox = Rect(tl, br);
    }

    return results;
}

}  // namespace mxutils

