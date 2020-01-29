#ifndef _MXNET_UTILS_UTILS_H_
#define _MXNET_UTILS_UTILS_H_

#include <map>
#include <string>

#include <mxnet-cpp/MxNetCpp.h>

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


}  // namespace mxutils

#endif
