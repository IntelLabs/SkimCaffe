#ifndef CAFFE_UTIL_MATH_FUNCTIONS_INTEL_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_INTEL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#include "SpMP/synk/barrier.hpp"

namespace caffe {

/** sparse convolution fused with bias term */
template <bool FUSE_RELU = false>
void caffe_cpu_sconv(
    // input features
    const float *input_padded, int in_channels,
    int height, int width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    // weights
    const int *rowptr, const int *colidx, const float *values,
    int kernel_h, int kernel_w,
    const int **rowptr_blocked, const int **colidx_blocked, const float **values_blocked,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int out_channels,
    float *output_scratch,
    int ninputs /* batch size*/);

/** sparse convolution fused with bias term, relu, and pooling layer */
template <typename Dtype>
void caffe_cpu_sconv_fused_with_relu_and_pooling(
    // input features
    const Dtype *input_padded, int in_channels,
    int height, int width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    // weights
    const int *rowptr, const int *colidx, const Dtype *values,
    int kernel_h, int kernel_w,
    // bias (for the case when bias is fused with convolution)
    const Dtype *bias,
    // pooling (for the case when pooling is fused with convolution)
    float *pool_top, int *mask,
    // output features
    Dtype *output,
    int out_channels,
    int ninputs);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_INTEL_H_
