#pragma once

namespace caffe
{

//template <typename Dtype>
//void caffe_gpu_impose_sparsity(Dtype *weight_inout, Dtype *mask, double impose_factor, const double *A, int M, int N, int repeat);

template <typename Dtype>
void caffe_gpu_impose_sparsity(
  Dtype *weight, double *weight_temp, double **weight_temp_ptr,
  const double *A, double *A_temp, double **A_temp_ptr,
  Dtype *mask, double impose_factor,
  int M, int N, int repeat);

}
