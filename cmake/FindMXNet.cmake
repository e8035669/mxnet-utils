# This will define the following variables:
#
#   MXNet_FOUND            - True if the system has the MXNet library
#   MXNet_INCLUDE_DIRS     - location of header files
#   MXNet_LIBRARIES        - location of library files

find_path(MXNet_INCLUDE_DIRS mxnet-cpp/MxNetCpp.h
    "/usr/include"
    "/usr/local/include"
    "$ENV{HOME}/.local/include"
    )
find_library(MXNet_LIBRARIES libmxnet.so
    "/usr/lib"
    "/usr/local/lib"
    "$ENV{HOME}/.local/lib"
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MXNet
    FOUND_VAR MXNet_FOUND
    REQUIRED_VARS MXNet_INCLUDE_DIRS MXNet_LIBRARIES
    )
