#include "utils.h"
#include "mxnet-cpp/op.h"
#include "mxnet-cpp/operator.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/symbol.h"
#include "opencv2/core/cvstd_wrapper.hpp"

#include <fstream>
#include <sstream>
#include <string>

using namespace std;

namespace mxutils {

std::string to_string(const mxnet::cpp::Context& ctx) {
    stringstream ss;
    switch (ctx.GetDeviceType()) {
        case mxnet::cpp::kCPU:
            ss << "CPU";
            break;
        case mxnet::cpp::kGPU:
            ss << "GPU";
            break;
        case mxnet::cpp::kCPUPinned:
            ss << "CPUPinned";
    }
    ss << '(' << ctx.GetDeviceId() << ')';
    return ss.str();
}

std::string shapeof(const mxnet::cpp::NDArray& array) {
    std::vector<mx_uint> arrayShape = array.GetShape();
    stringstream ss;
    ss << '(';
    for (size_t i = 0; i < arrayShape.size(); ++i) {
        if (i) ss << ", ";
        ss << arrayShape[i];
    }
    ss << ios::dec;
    ss << ")";
    return ss.str();
}

std::string to_string(const mxnet::cpp::NDArray& array) {
    return shapeof(array) + " @" + to_string(array.GetContext());
}

void splitParamMap(const NDArrayMap& paramMap,
                   NDArrayMap* argParamInTargetContext,
                   NDArrayMap* auxParamInTargetContext,
                   mxnet::cpp::Context targetContext) {
    for (const auto& p : paramMap) {
        string type = p.first.substr(0, 4);
        string name = p.first.substr(4);
        if (type == "arg:") {
            (*argParamInTargetContext)[name] = p.second.Copy(targetContext);
        } else if (type == "aux:") {
            (*auxParamInTargetContext)[name] = p.second.Copy(targetContext);
        }
    }
}

mxnet::cpp::NDArray asInContext(const mxnet::cpp::NDArray& array,
                                const mxnet::cpp::Context& ctx) {
    if (array.GetContext().GetDeviceId() == ctx.GetDeviceId() &&
        array.GetContext().GetDeviceType() == ctx.GetDeviceType()) {
        return array;
    } else {
        return array.Copy(ctx);
    }
}

std::vector<std::string> loadLineFromIstream(std::istream& is) {
    std::vector<std::string> context;
    for (std::string s; std::getline(is, s);) {
        context.push_back(s);
    }
    return context;
}

std::vector<std::string> loadLinesInFile(const std::string& path) {
    ifstream ifs(path);
    return loadLineFromIstream(ifs);
}

std::vector<std::string> loadClassnames(const std::string& path) {
    return loadLinesInFile(path);
}

void convertMat2NDArrayCHW(cv::InputArray input, mxnet::cpp::NDArray* output) {
    convertMat2NDArrayHWC(input, output);
    if (output != nullptr) {
        mxnet::cpp::Operator("transpose")(mxnet::cpp::Shape(0, 3, 1, 2),
                                          *output)
            .Invoke(*output);
    }
}

void convertMat2NDArrayHWC(cv::InputArray input, mxnet::cpp::NDArray* output) {
    if (output != nullptr) {
        cv::Mat inputMat = input.getMat();
        cv::Mat floatMat;
        inputMat.convertTo(floatMat, CV_32F, 1 / 255.);
        size_t elems = floatMat.cols * floatMat.rows * floatMat.channels();
        output->SyncCopyFromCPU((float*)floatMat.data, elems);
    }
}

}  // namespace mxutils

