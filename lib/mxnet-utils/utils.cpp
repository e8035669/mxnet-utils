#include "utils.h"

#include <sstream>

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

}  // namespace mxutils

