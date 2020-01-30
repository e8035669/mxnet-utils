#ifndef _MXNET_UTILS_UTILS_H_
#define _MXNET_UTILS_UTILS_H_

#include <map>
#include <string>

#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/core.hpp>

namespace mxutils {

typedef std::map<std::string, mxnet::cpp::NDArray> NDArrayMap;

std::string to_string(const mxnet::cpp::Context& ctx);

std::string shapeof(const mxnet::cpp::NDArray& array);

std::string to_string(const mxnet::cpp::NDArray& array);

void splitParamMap(const NDArrayMap& paramMap,
                   NDArrayMap* argParamInTargetContext,
                   NDArrayMap* auxParamInTargetContext,
                   mxnet::cpp::Context targetContext);

mxnet::cpp::NDArray asInContext(const mxnet::cpp::NDArray& array,
                                const mxnet::cpp::Context& ctx);

void convertMat2NDArray(cv::InputArray input, mxnet::cpp::NDArray* output);

void convertMat2NDArrayHWC(cv::InputArray input, mxnet::cpp::NDArray* output);

void convertNDArray2Mat(const mxnet::cpp::NDArray& input,
                        cv::OutputArray output);

void convertNDArray2MatHWC(const mxnet::cpp::NDArray& input,
                        cv::OutputArray output);


}  // namespace mxutils

#endif
