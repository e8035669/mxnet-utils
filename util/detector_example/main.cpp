#include <chrono>

#include <clipp.h>
#include <mxnet-cpp/MxNetCpp.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

#include "detect_result.h"
#include "draw.h"

using namespace std;
using namespace cv;
using namespace mxnet::cpp;
using namespace mxutils;
using namespace clipp;

vector<string> loadLines(const std::string& path) {
    ifstream ifs(path);
    vector<string> context;
    for (string s; getline(ifs, s);) {
        context.push_back(s);
    }
    return context;
}

int main(int argc, char** argv) {
    string modelPath, parameterPath, inputShapeStr;
    string videoPath, clsNamePath;
    vector<string> picPaths;
    enum class Mode { VIDEO, PIC };
    Mode mode = Mode::PIC;

    auto cli =
        (value("model path", modelPath).doc("json file path of your model"),  //
         value("parameter path", parameterPath)
             .doc("weights of your model"),  //
         value("input shape", inputShapeStr)
             .doc("Input shape of your first layer"),  //
         option("-v", "--video")
                 .set(mode, Mode::VIDEO)
                 .doc("Load each frame in this video to predict") &
             value("video path", videoPath),
         option("-p", "--pic")
                 .set(mode, Mode::PIC)
                 .doc("Load each image to predict") &
             values("pic paths", picPaths),
         option("-n", "--names").doc(".names file contains class names") &
             value("name file", clsNamePath));

    if (!parse(argc, argv, cli)) {
        cout << make_man_page(cli, "detector_example");
        return -1;
    }

    int gpuCount = 0;
    MXGetGPUCount(&gpuCount);

    Context ctx = Context::cpu();
    if (gpuCount) ctx = Context::gpu();

    DetectPredictor predictor(ctx);
    predictor.loadModel(modelPath);
    predictor.loadParameter(parameterPath);

    Shape inputShape;
    stringstream ss(inputShapeStr);
    ss >> inputShape;
    predictor.setInputSize(inputShape);

    predictor.prepare();

    vector<string> names;
    if (!clsNamePath.empty()) {
        names = loadLines(clsNamePath);
    }

    bool run = true;

    if (mode == Mode::VIDEO) {
        VideoCapture cap(videoPath);
        Mat img;
        while (run) {
            cap >> img;
            vector<DetectResult> result = predictor.predict(img, 0.5);
            for (auto& r : result) {
                if (names.size() > 0) {
                    drawBbox(img, r, names);
                } else {
                    drawBbox(img, r);
                }
            }
            imshow(videoPath, img);
            char keyin = waitKey(30);
            if (keyin == 27) {
                run = false;
            }
            if (keyin == 's') {
                auto now = chrono::system_clock::now();
                chrono::nanoseconds nanos(now.time_since_epoch());
                string filename = "Image" + to_string(nanos.count()) + ".png";
                imwrite(filename, img);
                cout << "Save to '" << filename << "'." << endl;
            }
        }
    } else if (mode == Mode::PIC) {
        for (size_t i = 0; i < picPaths.size() && run; ++i) {
            Mat img = imread(picPaths[i]);
            vector<DetectResult> result = predictor.predict(img, 0.5);
            for (auto& r : result) {
                if (names.size() > 0) {
                    drawBbox(img, r, names);
                } else {
                    drawBbox(img, r);
                }
            }
            imshow("img", img);
            char keyin = waitKey(0);
            if (keyin == 27) {
                run = false;
            }
            if (keyin == 's') {
                auto now = chrono::system_clock::now();
                chrono::nanoseconds nanos(now.time_since_epoch());
                string filename = "Image" + to_string(nanos.count()) + ".png";
                imwrite(filename, img);
                cout << "Save to '" << filename << "'." << endl;
            }
        }
    }

    return 0;
}
