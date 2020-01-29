#include "classify_result.h"

using namespace mxnet::cpp;

namespace mxutils {

int toClassifyResult(const std::vector<mxnet::cpp::NDArray>& outputs) {
    if (outputs.size() < 1) {
        throw std::runtime_error("No outputs");
    }
    if (outputs[0].GetContext().GetDeviceType() != mxnet::cpp::kCPU) {
        throw std::runtime_error("Not in cpu");
    }
    NDArray output = outputs[0];
    NDArray argMax = output.ArgmaxChannel();
    argMax.WaitToRead();
    return argMax.At(0);
}

}  // namespace mxutils

