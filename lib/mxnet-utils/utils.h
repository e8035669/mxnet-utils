#ifndef _MXNET_UTILS_UTILS_H_
#define _MXNET_UTILS_UTILS_H_

#include <map>
#include <string>
#include <vector>

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

std::vector<std::string> loadLineFromIstream(std::istream& is);

std::vector<std::string> loadLinesInFile(const std::string& path);

std::vector<std::string> loadClassnames(const std::string& path);

void convertMat2NDArrayCHW(cv::InputArray input, mxnet::cpp::NDArray* output);

void convertMat2NDArrayHWC(cv::InputArray input, mxnet::cpp::NDArray* output);

}  // namespace mxutils

#endif
