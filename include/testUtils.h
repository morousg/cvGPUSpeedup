#pragma once

#include <opencv2/core.hpp>
#include <string>

template <int Depth>
std::string depthToString() { return ""; }

#define DEPTH_TO_STRING(cv_depth, string_t) \
template <>                                 \
std::string depthToString<cv_depth>() {     \
    return string_t;                        \
}

DEPTH_TO_STRING(CV_8U, "CV_8U")
DEPTH_TO_STRING(CV_8S, "CV_8S")
DEPTH_TO_STRING(CV_16U, "CV_16U")
DEPTH_TO_STRING(CV_16S, "CV_16S")
DEPTH_TO_STRING(CV_32S, "CV_32S")
DEPTH_TO_STRING(CV_32F, "CV_32F")
DEPTH_TO_STRING(CV_64F, "CV_64F")

template <int Channels>
std::string channelsToString() { return ""; }

#define CHANNELS_TO_STRING(cv_channels, string_t) \
template <>                                       \
std::string channelsToString<cv_channels>() {     \
    return string_t;                              \
}

CHANNELS_TO_STRING(1, "C1")
CHANNELS_TO_STRING(2, "C2")
CHANNELS_TO_STRING(3, "C3")
CHANNELS_TO_STRING(4, "C4")

template <int T>
std::string cvTypeToString() {
    return depthToString<CV_MAT_DEPTH(T)>() + channelsToString<CV_MAT_CN(T)>();
}