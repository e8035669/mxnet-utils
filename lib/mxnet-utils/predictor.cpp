#include "predictor.h"

#include "utils.h"


using namespace mxnet::cpp;

namespace mxutils {

Predictor::Predictor(const mxnet::cpp::Context& ctx) : ctx(ctx) {}

Predictor::~Predictor() {}

void Predictor::loadModel(const std::string& filename) {
    this->net = Symbol::Load(filename);
}

void Predictor::loadParameter(const std::string& filename) {
    NDArrayMap params;
    NDArray::Load(filename, nullptr, &params);
    splitParamMap(params, &argParam, &auxParam, ctx);
}

void Predictor::setInputSize(const mxnet::cpp::Shape& shape) {
    this->inputShape = shape;
}

mxnet::cpp::Shape Predictor::getInputSize() {
    return inputShape;
}

void Predictor::prepare() {
    argParam["data"] = NDArray(inputShape, ctx);
    exec.reset(net.SimpleBind(ctx, argParam, {}, {}, auxParam));
}

std::vector<mxnet::cpp::NDArray> Predictor::predict(
    const mxnet::cpp::NDArray& data) {
    data.CopyTo(&argParam["data"]);
    exec->Forward(false);
    std::vector<NDArray> out(exec->outputs.size());
    for (size_t i = 0; i < exec->outputs.size(); ++i) {
        out[i] = asInContext(exec->outputs[i], Context::cpu());
        // out[i].WaitToRead();
    }
    return out;
}

}  // namespace mxutils

