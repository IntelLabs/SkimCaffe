#pragma once

namespace caffe
{

template <typename Dtype>
void caffe_gpu_impose_sparsity(Dtype *weight_inout, Dtype *mask, double impose_factor, const double *A, int M, int N, int repeat);

}
