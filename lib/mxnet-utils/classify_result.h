#ifndef _MXUTILS_CLASSIFY_RESULT_H_
#define _MXUTILS_CLASSIFY_RESULT_H_

#include <vector>

#include <mxnet-cpp/MxNetCpp.h>

namespace mxutils {

int toClassifyResult(const std::vector<mxnet::cpp::NDArray>& outputs);
}

#endif
