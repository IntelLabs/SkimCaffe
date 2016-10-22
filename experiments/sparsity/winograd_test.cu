#include <assert.h>

#include <cuda.h>
#include <cusolverDn.h>

#include "caffe/util/math_functions.hpp"
#include "caffe/util/winograd.hpp"
#include "winograd_test.h"

template <typename Dtype>
__global__ void scan_kernel(Dtype *g_odata, const Dtype *g_idata, int n)
{
  __shared__ Dtype temp[2*64]; // allocated on invocation
  int thid = threadIdx.x;
  int pout = 0, pin = 1;
  // Load input into shared memory.
  // This is exclusive scan, so shift right by one
  // and set first element to 0
  temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
  __syncthreads();
  for (int offset = 1; offset < n; offset *= 2)
  {
   pout = 1 - pout; // swap double buffer indices
   pin = 1 - pout;
   if (thid >= offset)
     temp[pout*n+thid] = temp[pin*n+thid] + temp[pin*n+thid - offset];
   else
     temp[pout*n+thid] = temp[pin*n+thid];
   __syncthreads();
  }
  g_odata[thid] = temp[pout*n+thid]; // write output
}


// input matrix row major, output matrix col major when new_m >= n
template <typename Dtype>
__global__ void compact_kernel(double *out_matrix, const double *in_matrix, const Dtype *mask, int old_m, int new_m, int n)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  while (tid < old_m) {
    if (tid == old_m - 1 && mask[tid] != new_m || mask[tid + 1] > mask[tid]) {
      for (int j = 0; j < n; ++j) {
        out_matrix[j*new_m + ((int)mask[tid])] = in_matrix[tid*n + j];
      }
    }

    tid += gridDim.x*blockDim.x;
  }
}

template <typename Dtype>
void caffe_gpu_scan(Dtype *g_odata, const Dtype *g_idata, int n)
{
  scan_kernel<<<1, n>>>(g_odata, g_idata, n); 
}

template void caffe_gpu_scan<double>(double *g_odata, const double *g_idata, int n);
template void caffe_gpu_scan<float>(float *g_odata, const float *g_idata, int n);

template <typename Dtype>
void caffe_compact(double *out_matrix, const double *in_matrix, const Dtype *mask, int old_m, int new_m, int n)
{
  compact_kernel<<<1, old_m>>>(out_matrix, in_matrix, mask, old_m, new_m, n);
}

template
void caffe_compact<double>(double *out_matrix, const double *in_matrix, const double *mask, int old_m, int new_m, int n);
template
void caffe_compact<float>(double *out_matrix, const double *in_matrix, const float *mask, int old_m, int new_m, int n);

cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;
int cuda_lwork;
double *cuda_work;

double *A_temp, *S, *U, *V, *VT, *rwork;
int *dev_info;

template <typename Dtype>
void imposeSparsityGPU(Dtype *weight_inout_cpu, const Dtype *mask_cpu, const double *A, int M, int N)
{
  cusolverStatus_t cusolver_status;
  cublasStatus_t cublas_status;
  cudaError_t cudaStat;
  
  Dtype *weight_input;
  Dtype *mask, *mask2;
  
  cudaStat = cudaMalloc((void **)&weight_input, sizeof(Dtype)*N*N);
  assert(cudaSuccess == cudaStat);
  
  cudaStat = cudaMalloc((void **)&mask, sizeof(Dtype)*M*M);
  assert(cudaSuccess == cudaStat);
  
  cudaStat = cudaMalloc((void **)&mask2, sizeof(Dtype)*M*M);
  assert(cudaSuccess == cudaStat);
  
  cudaMemcpy(weight_input, weight_inout_cpu, sizeof(Dtype)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(mask, mask_cpu, sizeof(Dtype)*M*M, cudaMemcpyHostToDevice);

  if (NULL == cusolverH) {
  	fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
  	
    // step 1: create cusolverDn handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // step 2: copy data to device
    cudaStat = cudaMalloc((void **)&A_temp, sizeof(double)*(M*M)*(N*N));
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **)&S, sizeof(double)*(N*N));
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **)&U, sizeof(double)*(M*M)*(M*M));
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **)&VT, sizeof(double)*(M*M)*(M*M));
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **)&V, sizeof(double)*(N*N)*(N*N));
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **)&rwork, sizeof(double)*(N*N - 1));
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMalloc((void **)&dev_info, sizeof(int));
    assert(cudaSuccess == cudaStat);

    // step 3: query working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(
      cusolverH,
      M*M,
      N*N,
      &cuda_lwork);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cudaStat = cudaMalloc((void **)&cuda_work, sizeof(double)*cuda_lwork);
    assert(cudaSuccess == cudaStat);
  }

  Dtype zero_cnt_temp;
  caffe::caffe_gpu_if_zerout(M*M, mask, mask, (Dtype)0);
  caffe::caffe_gpu_asum(M*M, mask, &zero_cnt_temp);
  int zero_cnt = zero_cnt_temp;
  
  fprintf(stderr, "zero_cnt = %d\n", zero_cnt);
  
  Dtype mask_temp[M*M];
  cudaMemcpy(mask_temp, mask, sizeof(Dtype)*M*M, cudaMemcpyDeviceToHost);
  //print_matrix(mask_temp, M, M);
  //fprintf(stderr, "\n");
  
  scan_kernel<<<1, M*M>>>(mask, mask, M*M);
  
  cudaMemcpy(mask_temp, mask, sizeof(Dtype)*M*M, cudaMemcpyDeviceToHost);
  //print_matrix(mask_temp, M, M);
  //fprintf(stderr, "\n");
  
  compact_kernel<<<1, M*M>>>(A_temp, A, mask, M*M, (int)zero_cnt, N*N);
  
  Dtype A_temp_cpu[M*M*N*N];
  cudaMemcpy(A_temp_cpu, A_temp, sizeof(Dtype)*zero_cnt*N*N, cudaMemcpyDeviceToHost);
  
  //fprintf(stderr, "A_temp_T\n");
  //print_matrix(A_temp_cpu, N*N, zero_cnt);
  
  int rank = std::min(zero_cnt, N*N);

  signed char jobu = 'A';
  signed char jobvt = 'A';

	// gesvd only supports m >= n
	// gesvd only supports jobu = 'A' and jobvt = 'A'
  cusolver_status = cusolverDnDgesvd(
      cusolverH,
      jobu, jobvt,
      zero_cnt, N*N,
      A_temp, zero_cnt,
      S,
      U, zero_cnt,
      VT, N*N,
      cuda_work, cuda_lwork,
      NULL, dev_info);
  int dev_info_cpu;
  cudaMemcpy(&dev_info_cpu, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  fprintf(stderr, "dev_info = %d\n", dev_info_cpu);
  if (CUSOLVER_STATUS_SUCCESS != cusolver_status) {
  	fprintf(stderr, "cusolverDnDgesvd returns %d\n", cusolver_status);
  }
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  // U  is (M*M)x(rank) matrix
  // VT is (N*N)x(N*N) matrix, but we only need first rank rows
  
  fprintf(stderr, "V\n");
  Dtype VT_cpu[(N*N)*(N*N)];
  cudaMemcpy(VT_cpu, VT, sizeof(Dtype)*N*N*N*N, cudaMemcpyDeviceToHost);
  print_matrix(VT_cpu, N*N, N*N);

//  cublas_status = cublasDgeam(
//      cublasH,
//      CUBLAS_OP_T, CUBLAS_OP_N,
//      1, VT, rank,
//      0, VT, rank,
//      V, N*N);
//  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // compute V*V^T*W with (W^T*V*V^T)^T
  double alpha = 1;
  double beta = 0;
  double *W;
  cudaStat = cudaMalloc((void **)&W, sizeof(double)*(N*N)*(N*N));
  assert(cudaSuccess == cudaStat);
  
  cudaMemset(W, 0, sizeof(double)*(N*N)*(N*N));
  
  cublasDsyrk(
      cublasH,
      CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
      N*N, rank,
      &alpha, VT, N*N,
      &beta,
      W, N*N);
      
  Dtype W_cpu[(N*N)*(N*N)];
  cudaMemcpy(W_cpu, W, sizeof(Dtype)*(N*N)*(N*N), cudaMemcpyDeviceToHost);
  fprintf(stderr, "V_times_VT\n");
  print_matrix(W_cpu, N*N, N*N);

//
//
//  MKL_INT info = LAPACKE_dgesvd(
//    cusolverH,
//    LAPACK_ROW_MAJOR, 'S', 'S',
//    zero_cnt, N*N,
//    A_temp, N*N,
//    S,
//    U, rank,
//    VT, N*N,
//    superb);
//  if (info > 0) {
//    LOG(FATAL) << "SVD failed to converge with return value " << info;
//  }
//
////    fprintf(stderr, "eigenvalues: "); print_matrix(S, 1, rank);
//  int rank_truncated = rank;
//  for (int k = 0; k < rank; ++k) {
//    if (S[k] < 1e-5) {
//      rank_truncated = k;
//      break;
//    }
//  }
////    fprintf(stderr, "VT\n"); print_matrix(VT, rank, N*N);
//
//  Dtype weight_temp[N*N];
//  memcpy(weight_temp, weight_inout, N*N*sizeof(Dtype));
//
////    double V_times_VT[(N*N)*(N*N)];
////    fprintf(stderr, "V_times_VT\n");
//  for (int i = 0; i < N*N; ++i) {
//    double acc = weight_temp[i];
//    for (int j = 0; j < N*N; ++j) {
//      double sum = 0;
//      for (int k = 0; k < rank_truncated; ++k) {
//        sum += VT[k*N*N + i]*VT[k*N*N + j];
//      }
////        fprintf(stderr, "%g ", sum);
//      acc -= sum*weight_temp[j];
//    }
////      fprintf(stderr, "\n");
//    weight_inout[i] = acc;
//  }
}

template
void imposeSparsityGPU<double>(double *weight_inout, const double *mask, const double *A, int M, int N);

template
void imposeSparsityGPU<float>(float *weight_inout, const float *mask, const double *A, int M, int N);
