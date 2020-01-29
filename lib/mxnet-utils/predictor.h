#ifndef _MXNET_UTILS_CLASSIFIER_PREDICTOR_H_
#define _MXNET_UTILS_CLASSIFIER_PREDICTOR_H_

#include <map>
#include <memory>

#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/core.hpp>

namespace mxutils {

class Predictor {
   public:
    Predictor(const mxnet::cpp::Context& ctx);
    ~Predictor();

    void loadModel(const std::string& filename);

    void loadParameter(const std::string& filename);

    void setInputSize(const mxnet::cpp::Shape& shape);

    void prepare();

    std::vector<mxnet::cpp::NDArray> predict(const mxnet::cpp::NDArray& data);

   private:
    mxnet::cpp::Context ctx;
    mxnet::cpp::Symbol net;
    mxnet::cpp::Shape inputShape;
    std::map<std::string, mxnet::cpp::NDArray> argParam;
    std::map<std::string, mxnet::cpp::NDArray> auxParam;
    std::shared_ptr<mxnet::cpp::Executor> exec;
};

}  // namespace mxutils

#endif
