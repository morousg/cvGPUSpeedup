

#include "testUtils.h"
#include <cv2cuda_types.cuh>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>

template <int T>
bool checkResults(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& h_comparison1C) {
    cv::Mat h_comparison(NUM_ELEMS_Y, NUM_ELEMS_X, T);
    cv::Mat maxError(NUM_ELEMS_Y, NUM_ELEMS_X, T, 0.00001);
    cv::compare(h_comparison1C, maxError, h_comparison, cv::CMP_GT);

#ifdef CVGS_DEBUG
    for (int y=0; y<h_comparison1C.rows; y++) {
        for (int x=0; x<h_comparison1C.cols; x++) {
            CUDA_T(T) value = h_comparison1C.at<CUDA_T(T)>(y,x);
            if (value > 0.00001) {
                std::cout << "(" << x << "," << y << ")= " << value << ";" << std::endl;
            }
        }
        std::cout << std::endl;
    }
#endif
    
    int errors = cv::countNonZero(h_comparison);
    return errors == 0;
}

template <int T>
bool compareAndCheck(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& cvVersion, cv::Mat& cvGSVersion) {
    bool passed = true;
    cv::Mat diff = cv::abs(cvVersion - cvGSVersion);
    std::vector<cv::Mat> h_comparison1C(CV_MAT_CN(T));
    cv::split(diff, h_comparison1C);

    for (int i=0; i<CV_MAT_CN(T); i++) {
        passed &= checkResults<CV_MAT_DEPTH(T)>(NUM_ELEMS_X, NUM_ELEMS_Y, h_comparison1C.at(i));
    }
    return passed;
}