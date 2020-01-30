#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>

#include "detect_result.h"
#include "draw.h"

using namespace std;
using namespace cv;
using namespace mxnet::cpp;
using namespace mxutils;

int main(int argc, char** argv) {

    int gpuCount = 0;
    MXGetGPUCount(&gpuCount);

    Context ctx = Context::cpu();
    if (gpuCount) ctx = Context::gpu();

    DetectPredictor predictor(ctx);
    predictor.loadModel(argv[1]);
    predictor.loadParameter(argv[2]);

    Shape inputShape;
    stringstream ss(argv[3]);
    ss >> inputShape;
    predictor.setInputSize(inputShape);

    predictor.prepare();

    bool run = true;
    if (string(argv[4]) == "video") {
        VideoCapture cap(argv[5]);
        Mat img;
        while (run) {
            cap >> img;
            vector<DetectResult> result = predictor.predict(img, 0.5);
            for (auto& r : result) {
                drawBbox(img, r);
            }
            imshow(argv[5], img);
            char keyin = waitKey(30);
            if (keyin == 27) {
                run = false;
            }
        }
    } else {
        for (int i = 5; i < argc && run; ++i) {
            Mat img = imread(argv[i]);
            vector<DetectResult> result = predictor.predict(img, 0.5);
            for (auto& r : result) {
                drawBbox(img, r);
            }
            imshow("img", img);
            char keyin = waitKey(0);
            if (keyin == 27) {
                run = false;
            }
        }
    }

    return 0;
}
