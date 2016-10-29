#include <assert.h>

#include <cuda.h>

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
__global__ void compact_kernel(double *out_matrix, const double *in_matrix, const Dtype *mask, double *out_b, const Dtype *in_b, int old_m, int new_m, int n, double impose_factor)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  while (tid < old_m + n) {
    if (tid >= old_m) {
      for (int j = 0; j < n; ++j) {
        out_matrix[j*(new_m + n) + tid - old_m + new_m] = 0;
      }
      out_matrix[(tid - old_m)*(new_m + n) + tid - old_m + new_m] = 1;
    }
    else if (tid == old_m - 1 && mask[tid] != new_m || mask[tid + 1] > mask[tid]) {
      for (int j = 0; j < n; ++j) {
        out_matrix[j*(new_m + n) + ((int)mask[tid])] = impose_factor*in_matrix[tid*n + j];
      }
    }
    
    if (tid < new_m) {
      out_b[tid] = 0;
    }
    else if (tid < new_m + n) {
    	out_b[tid] = in_b[tid - new_m];
    }

    tid += gridDim.x*blockDim.x;
  }
}

cublasHandle_t cublasH = NULL;

template <typename Dtype>
void imposeSparsityGPU(Dtype *weight_inout, Dtype *mask, double impose_factor, const double *A, int M, int N)
{
	cublasStatus_t cublas_status;
	
	if (NULL == cublasH) {
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	}
  
  cudaError_t cudaStat;
  
  Dtype zero_cnt_temp;
  caffe::caffe_gpu_if_zerout(M*M, mask, mask, (Dtype)0);
  caffe::caffe_gpu_asum(M*M, mask, &zero_cnt_temp);
  int zero_cnt = zero_cnt_temp;
  
	double *A_temp, *b;
	cudaStat = cudaMalloc((void **)&A_temp, sizeof(double)*(M*M + N*N)*(N*N));
  assert(cudaSuccess == cudaStat);

  scan_kernel<<<1, M*M>>>(mask, mask, M*M);
  
  cudaStat = cudaMalloc((void **)&b, sizeof(double)*(zero_cnt + N*N));
  assert(cudaSuccess == cudaStat);
  compact_kernel<<<1, M*M + N*N>>>(A_temp, A, mask, b, weight_inout, M*M, (int)zero_cnt, N*N, impose_factor);
  
  double *A_array_cpu[] = { A_temp };
  double *b_array_cpu[] = { b };
  double **A_array, **b_array;
  cudaStat = cudaMalloc((void **)&A_array, sizeof(A_array_cpu));
  assert(cudaSuccess == cudaStat);
  cudaStat = cudaMalloc((void **)&b_array, sizeof(b_array_cpu));
  assert(cudaSuccess == cudaStat);
  cudaMemcpy(A_array, A_array_cpu, sizeof(A_array_cpu), cudaMemcpyHostToDevice);
  cudaMemcpy(b_array, b_array_cpu, sizeof(b_array_cpu), cudaMemcpyHostToDevice);
  
	int info;
	cublas_status = cublasDgelsBatched(
		cublasH, CUBLAS_OP_N,
		zero_cnt + N*N, N*N,
		1,
		A_array, zero_cnt + N*N,
		b_array, zero_cnt + N*N,
		&info,
		NULL,
		1);
		
  if (CUBLAS_STATUS_SUCCESS != cublas_status) {
  	fprintf(stderr, "cublasDgelsBatched returns %d\n", cublas_status);
  }
  if (0 != info) {
  	fprintf(stderr, "%dth parameter is invalid\n", info);
  }
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);
  
  cudaMemcpy(weight_inout, b, sizeof(double)*N*N, cudaMemcpyDeviceToDevice);

	cudaFree(A_temp);  
	cudaFree(b);
	
  cudaFree(A_array);
  cudaFree(b_array);
}

template
void imposeSparsityGPU<double>(double *weight_inout, double *mask, double impose_factor, const double *A, int M, int N);

template
void imposeSparsityGPU<float>(float *weight_inout, float *mask, double impose_factor, const double *A, int M, int N);