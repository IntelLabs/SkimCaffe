/*
 * spgemm.hpp
 *
 *  Created on: May 13, 2016
 *      Author: jpark103
 */

#ifndef CAFFE_UTIL_SPGEMM_HPP_
#define CAFFE_UTIL_SPGEMM_HPP_

#include <map>
#include <string>
#include <omp.h>
#include <immintrin.h>
#include "SpMP/synk/barrier.hpp"

struct CSR
{
  float *values;
  int *colidx;
  int *rowptr;
  int m, n;
};

extern std::map<std::string, CSR> layer2weight;
extern std::map<std::string, float *> layer2bottom;
extern std::map<std::string, float *> layer2bias;

extern unsigned long long conv_cycles_of_this_batch[1024*16];
extern unsigned long long reduce_cycles[1024*16];

static int spgemm_flops(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    int m)
{
  int flops = 0;
#pragma omp parallel for reduction(+:flops)
  for (int i = 0; i < m; ++i) {
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      flops += 2*(B_i[ja + 1] - B_i[ja]);
    }
  }
  return flops;
}

static void csrmultd(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C,
    int m, int n)
{
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = bias[j];
    }
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        C[i*n + jb] += a_entry*b_entry;
      }
    }
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = std::max<float>(0, C[i*n + j]);
    }
  }
}

static int csrmultd_fused_flops(
    const float *A,
    const float *B_data, const int *B_j, const int *B_i,
    const float *C_data, const int *C_j, const int *C_i,
    const float *D_data, const int *D_j, const int *D_i,
    const float *B_bias, const float *C_bias, const float *D_bias,
    float *E,
    int A_num_rows,
    int A_num_cols, int B_num_cols, int C_num_cols, int D_num_cols,
    float *B_temp_global, float *C_temp_global)
{
  int flops = 0;

#pragma omp parallel reduction(+:flops)
  {
    int tid = omp_get_thread_num();
    float *B_temp = B_temp_global + tid*B_num_cols;
    float *C_temp = C_temp_global + tid*C_num_cols;

#pragma omp for
    for (int i = 0; i < A_num_rows; ++i) {
      for (int j = 0; j < B_num_cols; ++j) {
        B_temp[j] = B_bias[j];
      }
      for (int j = 0; j < A_num_cols; ++j) {
        float a_entry = A[i*A_num_cols + j];
        if (a_entry == 0) continue;
        for (int k = B_i[j]; k < B_i[j + 1]; ++k) {
          B_temp[B_j[k]] += a_entry*B_data[k];
          flops += 2;
        }
      }

      for (int j = 0; j < C_num_cols; ++j) {
        C_temp[j] = C_bias[j];
      }
      for (int j = 0; j < B_num_cols; ++j) {
        float b_entry = B_temp[j];
        if (b_entry <= 0) continue;
        for (int k = C_i[j]; k < C_i[j + 1]; ++k) {
          C_temp[C_j[k]] += b_entry*C_data[k];
          flops += 2;
        }
      }

      for (int j = 0; j < D_num_cols; ++j) {
        E[i*D_num_cols + j] = D_bias[j];
      }
      for (int j = 0; j < C_num_cols; ++j) {
        float c_entry = C_temp[j];
        if (c_entry <= 0) continue;
        for (int k = D_i[j]; k < D_i[j + 1]; ++k) {
          E[i*D_num_cols + D_j[k]] += c_entry*D_data[k];
          flops += 2;
        }
      }
    }
  }

  return flops;
}

/* E = A*B*C*D */
static void csrmultd_fused(
    const float *A,
    const float *B_data, const int *B_j, const int *B_i,
    const float *C_data, const int *C_j, const int *C_i,
    const float *D_data, const int *D_j, const int *D_i,
    const float *B_bias, const float *C_bias, const float *D_bias,
    float *E,
    int A_num_rows,
    int A_num_cols, int B_num_cols, int C_num_cols, int D_num_cols,
    float *B_temp_global, float *C_temp_global)
{
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    float *B_temp = B_temp_global + tid*B_num_cols;
    float *C_temp = C_temp_global + tid*C_num_cols;

//    for (int j = 0; j < B_num_cols; ++j) {
//      B_temp[j] = B_bias[j];
//    }
//    for (int j = 0; j < C_num_cols; ++j) {
//      C_temp[j] = C_bias[j];
//    }

#pragma omp for
    for (int i = 0; i < A_num_rows; ++i) {
      for (int j = 0; j < B_num_cols; ++j) {
        B_temp[j] = B_bias[j];
      }
      for (int j = 0; j < A_num_cols; ++j) {
        float a_entry = A[i*A_num_cols + j];
        if (a_entry == 0) continue;
        for (int k = B_i[j]; k < B_i[j + 1]; ++k) {
          B_temp[B_j[k]] += a_entry*B_data[k];
        }
      }

      for (int j = 0; j < C_num_cols; ++j) {
        C_temp[j] = C_bias[j];
      }
      for (int j = 0; j < B_num_cols; ++j) {
        float b_entry = B_temp[j];
//        B_temp[j] = B_bias[j];
        if (b_entry <= 0) continue;
        for (int k = C_i[j]; k < C_i[j + 1]; ++k) {
          C_temp[C_j[k]] += b_entry*C_data[k];
        }
      }

      for (int j = 0; j < D_num_cols; ++j) {
        E[i*D_num_cols + j] = D_bias[j];
      }
      for (int j = 0; j < C_num_cols; ++j) {
        float c_entry = C_temp[j];
//        C_temp[j] = C_bias[j];
        if (c_entry <= 0) continue;
        for (int k = D_i[j]; k < D_i[j + 1]; ++k) {
          E[i*D_num_cols + D_j[k]] += c_entry*D_data[k];
        }
      }
    }
  }
}

static void spgemm(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C_data, int *C_j, int *C_i, int *cnnz,
    int m, int n, float *x)
{
  for (int j = 0; j < n; ++j) {
    x[j] = bias[j];
  }

  int nnz = 0;
  C_i[0] = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        x[jb] += a_entry*b_entry;
      }
    }

    for (int j = 0; j < n; ++j) {
      if (x[j] > 0) {
        C_j[nnz] = j;
        C_data[nnz] = x[j];
        ++nnz;
      }
      x[j] = bias[j];
    }
    C_i[i + 1] = nnz;
  }

  *cnnz = nnz;
}

static void csrmultd_csc(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C,
    int m, int n)
{
//  for (int i = 0; i < m; ++i) {
//    printf("%d:%d\n", i, A_i[i + 1] - A_i[i]);
//  }
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = bias[i];
    }
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        C[i*n + jb] += a_entry*b_entry;
      }
    }
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = std::max<float>(0, C[i*n + j]);
    }
  }
}

#ifdef __AVX512F__

#define VLEN (16)
#define SIMDFPTYPE __m512

#define _MM_SET1(a) _mm512_set1_ps(a)
#define _MM_FMADD(a, b, c) _mm512_fmadd_ps(a, b, c)
#define _MM_ADD(a, b) _mm512_add_ps(a, b)
#define _MM_MAX(a, b) _mm512_max_ps(a, b)
#define _MM_SETZERO() _mm512_setzero_ps()

#define _MM_LOAD(a) _mm512_load_ps(a)
#define _MM_STORE(a, b) _mm512_store_ps(a, b)

#elif __AVX2__

#define VLEN (8)
#define SIMDFPTYPE __m256

#define _MM_SET1(a) _mm256_set1_ps(a)
#define _MM_FMADD(a, b, c) _mm256_fmadd_ps(a, b, c)
#define _MM_ADD(a, b) _mm256_add_ps(a, b)
#define _MM_MAX(a, b) _mm256_max_ps(a, b)
#define _MM_SETZERO() _mm256_setzero_ps()

#define _MM_LOAD(a) _mm256_load_ps(a)
#define _MM_STORE(a, b) _mm256_store_ps(a, b)

#else

#define VLEN (4)
#define SIMDFPTYPE __m128

#define _MM_SET1(a) _mm_set1_ps(a)
#define _MM_FMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define _MM_ADD(a, b) _mm_add_ps(a, b)
#define _MM_MAX(a, b) _mm_max_ps(a, b)
#define _MM_SETZERO() _mm_setzero_ps()

#define _MM_LOAD(a) _mm_load_ps(a)
#define _MM_STORE(a, b) _mm_store_ps(a, b)

#endif

extern synk::Barrier *barriers[256];

// Compute sparse matrix times dense matrix fused with ReLU
//
// C = A*B, A is m*K, B is K*n matrices
// in fc6 of AlexNet, A is 4K*9K = 144 MB (in dense) and B is 4K*256 = 9 MB.
// In KNL, assuming 64 threads, each thread works on 64*9K = 2.25 MB.
// With a reasonable sparsity, A matrix fits in L2$.
// Per thread, C is 64 KB

static void __attribute__((noinline)) csrmm_fused_B_decomposed(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B,
    float *C,
    int M, int N, int K,
    const float *bias,
    float *C_scratch_global,
    int num_of_B_row_partitions,
    int num_of_B_col_partitions)
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    // threads are arranged in column-major way
    // so that the threads that will reduce the same data
    // together are located close with each other.
    int tid_row = tid%num_of_B_row_partitions;
    int tid_col = tid/num_of_B_row_partitions;

#ifdef __AVX512F__
    const int REG_BLOCK_SIZE = 4;
#define J_PREFETCH_DISTANCE (1)
#elif defined(__AVX2__)
    const int REG_BLOCK_SIZE = 8;
#else
    const int REG_BLOCK_SIZE = 8;
#endif
    SIMDFPTYPE sum[REG_BLOCK_SIZE*2];

    int k_per_col_block = (N + num_of_B_col_partitions - 1)/num_of_B_col_partitions;
    int k_begin = std::min(k_per_col_block*tid_col, N);
    int k_end = std::min(k_begin + k_per_col_block, N);

    assert(k_per_col_block == REG_BLOCK_SIZE*VLEN);
    int i_block = M;
    float *C_scratch = C_scratch_global + tid*i_block*REG_BLOCK_SIZE*VLEN;

    for (int ii = 0; ii < M; ii += i_block) {
      unsigned long long t = __rdtsc();

      for (int i = ii; i < ii + i_block; ++i) {
#pragma unroll
        for (int k = 0; k < REG_BLOCK_SIZE*2; ++k) {
          sum[k] = _MM_SETZERO();
  //          _mm_prefetch((const char *)(C + i*n + kk + k*VLEN), _MM_HINT_T0);
        }

        int j_begin = A_i[tid_row*M + i], j_end = A_i[tid_row*M + i + 1];
#ifdef UNROLL
        int len = j_end - j_begin;
        int rem = len%16;
        for (int j = j_begin; j < j_begin + len - rem; j += 16) {
          int c;
          SIMDFPTYPE v_v;
          __m512i c_v = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i *)(A_j + j)));
          c_v = _mm512_mullo_epi32(c_v, _mm512_set1_epi32(N));
          __declspec(aligned(64)) int temp_c[16];
          _mm512_store_epi32(temp_c, c_v);
//          for (int l = 0; l < 16; ++l) {
//            if (temp_c[l]*N != A_j[j + l]*N) {
//              printf("expected %d actual %d\n", A_j[j + l]*N, temp_c[l]*N);
//              exit(-1);
//            }
//          }

          c = temp_c[0];
          v_v = _MM_SET1(A_data[j]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[1];
          v_v = _MM_SET1(A_data[j + 1]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[2];
          v_v = _MM_SET1(A_data[j + 2]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[3];
          v_v = _MM_SET1(A_data[j + 3]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[4];
          v_v = _MM_SET1(A_data[j + 4]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[5];
          v_v = _MM_SET1(A_data[j + 5]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[6];
          v_v = _MM_SET1(A_data[j + 6]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[7];
          v_v = _MM_SET1(A_data[j + 7]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[8];
          v_v = _MM_SET1(A_data[j + 8]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[9];
          v_v = _MM_SET1(A_data[j + 9]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[10];
          v_v = _MM_SET1(A_data[j + 10]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[11];
          v_v = _MM_SET1(A_data[j + 11]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[12];
          v_v = _MM_SET1(A_data[j + 12]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[13];
          v_v = _MM_SET1(A_data[j + 13]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }

          c = temp_c[14];
          v_v = _MM_SET1(A_data[j + 14]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
          }

          c = temp_c[15];
          v_v = _MM_SET1(A_data[j + 15]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k + REG_BLOCK_SIZE] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k + REG_BLOCK_SIZE]);
          }
        }

        for (int j = j_begin + len - rem; j < j_end; ++j) {
          int c;
          SIMDFPTYPE v_v;

          c = A_j[j];
          v_v = _MM_SET1(A_data[j]);
#pragma unroll(4)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c*N + k_begin + k*VLEN), sum[k]);
          }
        }
#else
        for (int j = j_begin; j < j_end; ++j) {
          int c = A_j[j];
          SIMDFPTYPE v_v = _MM_SET1(A_data[j]);
#pragma unroll
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            if (sizeof(A_j[0]) == 2) {
              sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c*N + k_begin + k*VLEN), sum[k]);
            }
            else {
              sum[k] = _MM_FMADD(v_v, _MM_LOAD(B + c + k_begin + k*VLEN), sum[k]);
            }
          }
        }
#endif

#pragma unroll
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          _MM_STORE(C_scratch + ((i - ii)*REG_BLOCK_SIZE + k)*VLEN, _MM_ADD(sum[k], sum[k + REG_BLOCK_SIZE]));
        }
      } // for each row

      conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;

      barriers[tid_col]->wait(tid_row);

      t = __rdtsc();

      int i_per_thread = (i_block + num_of_B_row_partitions - 1)/num_of_B_row_partitions;
      int i_begin = std::min(ii + i_per_thread*tid_row, ii + i_block);
      int i_end = std::min(i_begin + i_per_thread, ii + i_block);

      for (int i = i_begin; i < i_end; ++i) {
#pragma unroll
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          sum[k] = _MM_SET1(bias[i]);
        }
        for (int j = 0; j < num_of_B_row_partitions; ++j) {
#pragma unroll
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _MM_ADD(
                sum[k],
                _MM_LOAD(C_scratch_global + (((tid_col*num_of_B_row_partitions + j)*i_block + (i - ii))*REG_BLOCK_SIZE + k)*VLEN));
          }
        }
#pragma unroll
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          _MM_STORE(C + i*N + k_begin + k*VLEN, _MM_MAX(_MM_SETZERO(), sum[k]));
        }
      }

      reduce_cycles[tid*16] += __rdtsc() - t;

      barriers[tid_col]->wait(tid_row);
    }
  } // omp parallel
}

#ifdef __AVX512F__
#define CSRMM_REG_BLOCK_SIZE (4)
#elif defined(__AVX2__)
#define CSRMM_REG_BLOCK_SIZE (8)
#else
#define CSRMM_REG_BLOCK_SIZE (8)
#endif

// Compute sparse matrix times dense matrix fused with ReLU
//
// Use 2D-decomposition of matrix C
// Compute C = A*B, A is M*K, B is K*N matrices
// in fc6 of AlexNet, A is 4K*9K*4B = 144MB in dense and 28.8MB in sparse assuming sparsity p = 0.1,
// and B is 4K*256*4B = 9MB and C is 4MB.
// C is decomposed into a (P=num_of_C_row_partitions)*(Q=num_of_C_col_partitions) 2D grid
//
// For example, when C is decomposed into 16*4 with 64 threads,
// each thread reads 28.8MB/16 = 1.8MB portion of A,
// reads 9MB/4 = 2.25MB portion of B,
// writes 4MB/(4*16) = 64KB portion of C.
//
// Even assuming A, B, and C all fit in L2$ of KNL (which is not the case),
// FLOP/Byte = (p*M*K*N*2/P/Q)/(p*M*K*8/P + K*N*4/Q + M*N*4/P/Q)
// With M=4K, K=9K, N=256, P=16, Q=4, p=0.1,
// FLOP/Byte = 7 which is bandwidth bound
//
// Since A and B don't fit in L2$ of KNL, we block columns of A
//
// assumptions: omp_get_num_threads()%num_of_C_col_partitions == 0

//#define DBG_CSRMM
#ifdef DBG_CSRMM
#define ROW_TO_DEBUG (38)
#define COL_TO_DEBUG (17)
#endif

static void /*__attribute__((noinline))*/ csrmm_fused_C_decomposed(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B,
    float *C,
    int M, int N, int K,
    const float *bias,
    int num_of_C_col_partitions,
    int num_of_A_col_blocks)
{
  int nthreads = omp_get_max_threads()/num_of_C_col_partitions*num_of_C_col_partitions;

#pragma omp parallel num_threads(nthreads)
  {
    unsigned long long t = __rdtsc();

    assert(nthreads%num_of_C_col_partitions == 0);
    int num_of_C_row_partitions = nthreads/num_of_C_col_partitions;
    int tid = omp_get_thread_num();

    // threads are arranged in row-major way
    // so that the threads close with each other will access
    // the same A portion with constructive sharing.
    int tid_row = tid/num_of_C_col_partitions;
    int tid_col = tid%num_of_C_col_partitions;

    int i_per_thread = (M + num_of_C_row_partitions - 1)/num_of_C_row_partitions;
    int i_begin = std::min(i_per_thread*tid_row, M);
    int i_end = std::min(i_begin + i_per_thread, M);

    int k_per_thread = (N + num_of_C_col_partitions - 1)/num_of_C_col_partitions;
    int k_begin = std::min(k_per_thread*tid_col, N);
    int k_end = std::min(k_begin + k_per_thread, N);

#undef CSRMM_J_PREFETCH_DISTANCE
#define CSRMM_J_PREFETCH_DISTANCE (8)
    assert((k_end - k_begin) % CSRMM_REG_BLOCK_SIZE*VLEN == 0);

    SIMDFPTYPE acc[CSRMM_REG_BLOCK_SIZE/**2*/];
//#define CSRMM_REARRANGE_B
#ifdef CSRMM_REARRANGE_B
#ifdef __AVX512F__
#define CSRMM_REPLICATE_B
#endif
#ifdef CSRMM_REPLICATE_B
    const float *B_pr = B + (tid_col*num_of_C_row_partitions + tid_row)*K*k_per_thread;
#else
    const float *B_pr = B + tid_col*K*k_per_thread;
#endif
#else
    const float *B_pr = B + k_begin;
#endif
    float *C_pr = C + (tid_row*num_of_C_col_partitions + tid_col)*i_per_thread*k_per_thread;

    // first col block of A
    int A_col_block = 0;
    for (int i = i_begin; i < i_end; ++i) {
      for (int kk = 0; kk < k_per_thread/VLEN; kk += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
        for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) {
          acc[k] = _MM_SET1(bias[i]);
#ifdef DBG_CSRMM
          if (i == ROW_TO_DEBUG && k_begin + (kk + k)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
            printf("%g ", bias[ROW_TO_DEBUG]);
          }
#endif
          _mm_prefetch((const char *)(C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN), _MM_HINT_T0);
        }

#define CSRMM_FMADD(j) \
        _Pragma("unroll(CSRMM_REG_BLOCK_SIZE)") \
        for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) { \
          acc[k] = _MM_FMADD(_MM_SET1(A_data[j]), _MM_LOAD(B_pr + A_j[j] + (kk + k)*VLEN), acc[k]); \
          _mm_prefetch((const char *)(B_pr + A_j[j + CSRMM_J_PREFETCH_DISTANCE] + (kk + k)*VLEN), _MM_HINT_T0); \
          /*if (i == ROW_TO_DEBUG && k_begin + (kk + k)*VLEN == COL_TO_DEBUG/VLEN*VLEN) { \
            printf(" + %g*%d:%g", A_data[j], A_j[j]/N, B[A_j[j] + k_begin + (kk + k)*VLEN + COL_TO_DEBUG%VLEN]); \
          } \*/ \
        }

#define CSRMM_UNROLL_FACTOR (4)

#define CSRMM_INNER_PROD \
        int j_begin = A_i[num_of_A_col_blocks*i_begin + A_col_block*(i_end - i_begin) + (i - i_begin)]; \
        int j_end = A_i[num_of_A_col_blocks*i_begin + A_col_block*(i_end - i_begin) + (i - i_begin) + 1]; \
        int len = j_end - j_begin; \
        int rem = len%CSRMM_UNROLL_FACTOR; \
        for (int j = j_begin; j < j_begin + len - rem; j += CSRMM_UNROLL_FACTOR) { \
          /*_mm_prefetch((const char *)(A_data + j + 16), _MM_HINT_T0);*/ \
          /*_mm_prefetch((const char *)(A_j + j + 16), _MM_HINT_T0);*/ \
          CSRMM_FMADD(j); \
          CSRMM_FMADD(j + 1); \
          CSRMM_FMADD(j + 2); \
          CSRMM_FMADD(j + 3); \
        } \
        for (int j = j_begin + len - rem; j < j_end; ++j) { \
          CSRMM_FMADD(j); \
        }

        CSRMM_INNER_PROD;

#pragma unroll(CSRMM_REG_BLOCK_SIZE)
        for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) {
          _MM_STORE(C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN, acc[k]);
        }
      } // for each col block of C
    } // for each row

    for (A_col_block = 1; A_col_block < num_of_A_col_blocks - 1; ++A_col_block) {
      for (int i = i_begin; i < i_end; ++i) {
        for (int kk = 0; kk < k_per_thread/VLEN; kk += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) {
            acc[k] = _MM_LOAD(C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN);
          }

          CSRMM_INNER_PROD;

#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) {
            _MM_STORE(C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN, acc[k]);
          }
        } // for each col block of C
      } // for each row
    } // for each col block of A

    // last col block of A
    for (int i = i_begin; i < i_end; ++i) {
      for (int kk = 0; kk < k_per_thread/VLEN; kk += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
        for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) {
          acc[k] = _MM_LOAD(C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN);
        }

        CSRMM_INNER_PROD;

#undef CSRMM_INNER_PROD
#undef CSRMM_FMADD
#undef CSRMM_UNROLL_FACTOR
#undef CSRMM_J_PREFETCH_DISTANCE

#pragma unroll(CSRMM_REG_BLOCK_SIZE)
        for (int k = 0; k < CSRMM_REG_BLOCK_SIZE; ++k) {
          _MM_STORE(C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN, _MM_MAX(_MM_SETZERO(), acc[k]));
#ifdef DBG_CSRMM
          if (i == ROW_TO_DEBUG && k_begin + (kk + k)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
            float temp[VLEN];
            _MM_STORE(temp, acc[k]);
            printf(" = %g->%g:%d\n",
                temp[COL_TO_DEBUG%VLEN],
                C_pr[(i - i_begin)*k_per_thread + (kk + k)*VLEN + COL_TO_DEBUG%VLEN],
                C_pr + (i - i_begin)*k_per_thread + (kk + k)*VLEN + COL_TO_DEBUG%VLEN - C);
          }
#endif
        }
      } // for each col block of C
    } // for each row

    conv_cycles_of_this_batch[tid*16] = __rdtsc() - t;
  } // omp parallel
}

static void __attribute__((noinline)) csrmm(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B,
    float *C,
    int m, int n, int k,
    const float *bias,
    int col_block_size)
{
  int ncolblocks = k/col_block_size;

#pragma omp parallel
  {
#ifdef __AVX512F__
    const int REG_BLOCK_SIZE = 16;
    __m512 sum[REG_BLOCK_SIZE];
#elif defined(__AVX2__)
    const int REG_BLOCK_SIZE = 8;
    __m256 sum[REG_BLOCK_SIZE];
#else
    const int REG_BLOCK_SIZE = 8;
    __m128 sum[REG_BLOCK_SIZE];
#endif

    int cb = 0;
#pragma omp for nowait
    for (int i = 0; i < m; ++i) {
#ifdef __AVX512F__
      for (int kk = 0; kk < n; kk += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 256
#pragma unroll(16)
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          sum[k] = _mm512_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m512 v_v = _mm512_set1_ps(A_data[j]);
#pragma unroll(16)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _mm512_fmadd_ps(v_v, _mm512_load_ps(B + c + kk + k*VLEN), sum[k]);
          }
        }
#pragma unroll(16)
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          _mm512_store_ps(C + i*n + kk + k*VLEN, sum[k]);
        }
      }
#elif defined(__AVX2__)
      for (int kk = 0; kk < n; kk += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 64
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          sum[k] = _mm256_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m256 v_v = _mm256_set1_ps(A_data[j]);
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*VLEN), sum[k]);
          }
        }
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          _mm256_store_ps(C + i*n + kk + k*VLEN, sum[k]);
        }
      }
#else
      for (int kk = 0; kk < n; kk += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 32
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          sum[k] = _mm_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m128 v_v = _mm_set1_ps(A_data[j]);
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*VLEN)), sum[k]);
          }
        }
        for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
          _mm_store_ps(C + i*n + kk + k*VLEN, sum[k]);
        }
      }
#endif
    }

    for (cb = 1; cb < ncolblocks; ++cb) {
#pragma omp for nowait
      for (int i = 0; i < m; ++i) {
#ifdef __AVX512F__
        for (int kk = 0; kk < n; kk += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 64
#pragma unroll(16)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _mm512_load_ps(C + i*n + kk + k*VLEN);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m512 v_v = _mm512_set1_ps(A_data[j]);
#pragma unroll(16)
            for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
              sum[k] = _mm512_fmadd_ps(v_v, _mm512_load_ps(B + c + kk + k*VLEN), sum[k]);
            }
          }
#pragma unroll(16)
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            _mm512_store_ps(C + i*n + kk + k*VLEN, sum[k]);
          }
        }
#elif defined(__AVX2__)
        for (int kk = 0; kk < n; kk += REG_BLOCK_SIZE*VLEN) {
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _mm256_load_ps(C + i*n + kk + k*VLEN);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m256 v_v = _mm256_set1_ps(A_data[j]);
            for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
              sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*VLEN), sum[k]);
            }
          }
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            _mm256_store_ps(C + i*n + kk + k*VLEN, sum[k]);
          }
        }
#else
        for (int kk = 0; kk < n; kk += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 32
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            sum[k] = _mm_load_ps(C + i*n + kk + k*VLEN);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m128 v_v = _mm_set1_ps(A_data[j]);
            for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
              sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*VLEN)), sum[k]);
            }
          }
          for (int k = 0; k < REG_BLOCK_SIZE; ++k) {
            _mm_store_ps(C + i*n + kk + k*VLEN, sum[k]);
          }
        }
#endif
      }
    } // for each col block
  } // omp parallel
}

static void spgemm_csc(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C_data, int *C_j, int *C_i, int *cnnz,
    int m, int n, float *x)
{
  for (int j = 0; j < n; ++j) {
    x[j] = 0;
  }

  int nnz = 0;
  C_i[0] = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        x[jb] += a_entry*b_entry;
      }
    }

    for (int j = 0; j < n; ++j) {
      if (bias[i] + x[j] > 0) {
        C_j[nnz] = j;
        C_data[nnz] = bias[i] + x[j];
        ++nnz;
      }
      x[j] = 0;
    }
    C_i[i + 1] = nnz;
  }

  *cnnz = nnz;
}

#endif /* CAFFE_UTIL_SPGEMM_HPP_ */
