/*
 * spgemm.hpp
 *
 *  Created on: May 13, 2016
 *      Author: jpark103
 */

#ifndef _CAFFE_UTIL_SPGEMM_HPP_
#define _CAFFE_UTIL_SPGEMM_HPP_

#include <map>
#include <string>
#include <omp.h>
#include <immintrin.h>
#include "intrinsic.hpp"

extern unsigned long long conv_cycles_of_this_batch[1024*16];
extern unsigned long long reduce_cycles[1024*16];

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
  int nnz = A_i[M*num_of_A_col_blocks];

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

    int m_per_thread = (M + num_of_C_row_partitions - 1)/num_of_C_row_partitions;
    int m_begin = std::min(m_per_thread*tid_row, M);
    int m_end = std::min(m_begin + m_per_thread, M);

    int n_per_thread = (N + num_of_C_col_partitions - 1)/num_of_C_col_partitions;
    int n_begin = std::min(n_per_thread*tid_col, N);
    int n_end = std::min(n_begin + n_per_thread, N);

#undef CSRMM_J_PREFETCH_DISTANCE
#define CSRMM_J_PREFETCH_DISTANCE (8)
    assert((n_end - n_begin) % VLEN == 0);

    SIMDFPTYPE acc[CSRMM_REG_BLOCK_SIZE/**2*/];
//#define CSRMM_REARRANGE_B
#ifdef CSRMM_REARRANGE_B
#ifdef __AVX512F__
#define CSRMM_REPLICATE_B
#endif
#ifdef CSRMM_REPLICATE_B
    const float *B_pr = B + (tid_col*num_of_C_row_partitions + tid_row)*K*n_per_thread;
#else
    const float *B_pr = B + tid_col*K*n_per_thread;
#endif
#else
    const float *B_pr = B + n_begin;
#endif
    float *C_pr = C + (tid_row*num_of_C_col_partitions + tid_col)*m_per_thread*n_per_thread;

    int A_col_block = 0;
    if (num_of_A_col_blocks > 1) {
      // first col block of A
      for (int m = m_begin; m < m_end; ++m) {
        int nn;
        for (nn = 0; nn < n_per_thread/VLEN/CSRMM_REG_BLOCK_SIZE*CSRMM_REG_BLOCK_SIZE; nn += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
            acc[n] = _MM_SET1(bias[m]);
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              printf("%g ", bias[ROW_TO_DEBUG]);
            }
#endif
            _mm_prefetch((const char *)(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN), _MM_HINT_T0);
          }

#define CSRMM_FMADD(j) \
          _Pragma("unroll(CSRMM_REG_BLOCK_SIZE)") \
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) { \
            acc[n] = _MM_FMADD(_MM_SET1(A_data[j]), _MM_LOAD(B_pr + A_j[j] + (nn + n)*VLEN), acc[n]); \
            assert(j < nnz); \
            assert(B_pr - B + A_j[j] + (nn + n + 1)*VLEN <= K*N); \
            _mm_prefetch((const char *)(B_pr + A_j[j + CSRMM_J_PREFETCH_DISTANCE] + (nn + n)*VLEN), _MM_HINT_T0); \
            /*if (i == ROW_TO_DEBUG && k_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) { \
              printf(" + %g*%d:%g", A_data[j], A_j[j]/N, B[A_j[j] + k_begin + (nn + n)*VLEN + COL_TO_DEBUG%VLEN]); \
            } \*/ \
          }

#define CSRMM_UNROLL_FACTOR (4)

#define CSRMM_INNER_PROD \
          int j_begin = A_i[num_of_A_col_blocks*m_begin + A_col_block*(m_end - m_begin) + (m - m_begin)]; \
          int j_end = A_i[num_of_A_col_blocks*m_begin + A_col_block*(m_end - m_begin) + (m - m_begin) + 1]; \
          assert(num_of_A_col_blocks*m_begin + A_col_block*(m_end - m_begin) + (m - m_begin) < M*num_of_A_col_blocks); \
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
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
            _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, acc[n]);
          }
        } // for each col block of C

        // remainder col block of C
        if (nn < n_per_thread/VLEN) {
          int n_rem = n_per_thread/VLEN - nn;
          for (int n = 0; n < n_rem; ++n) {
            acc[n] = _MM_SET1(bias[m]);
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              printf("%g ", bias[ROW_TO_DEBUG]);
            }
#endif
            _mm_prefetch((const char *)(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN), _MM_HINT_T0);
          }

#define CSRMM_FMADD_WO_UNROLL(j) \
          for (int n = 0; n < n_rem; ++n) { \
            acc[n] = _MM_FMADD(_MM_SET1(A_data[j]), _MM_LOAD(B_pr + A_j[j] + (nn + n)*VLEN), acc[n]); \
            _mm_prefetch((const char *)(B_pr + A_j[j + CSRMM_J_PREFETCH_DISTANCE] + (nn + n)*VLEN), _MM_HINT_T0); \
            /*if (i == ROW_TO_DEBUG && k_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) { \
              printf(" + %g*%d:%g", A_data[j], A_j[j]/N, B[A_j[j] + k_begin + (nn + n)*VLEN + COL_TO_DEBUG%VLEN]); \
            } \*/ \
          }

#define CSRMM_INNER_PROD_WO_UNROLL \
          int j_begin = A_i[num_of_A_col_blocks*m_begin + A_col_block*(m_end - m_begin) + (m - m_begin)]; \
          int j_end = A_i[num_of_A_col_blocks*m_begin + A_col_block*(m_end - m_begin) + (m - m_begin) + 1]; \
          assert(num_of_A_col_blocks*m_begin + A_col_block*(m_end - m_begin) + (m - m_begin) < M*num_of_A_col_blocks); \
          int len = j_end - j_begin; \
          int rem = len%CSRMM_UNROLL_FACTOR; \
          for (int j = j_begin; j < j_begin + len - rem; j += CSRMM_UNROLL_FACTOR) { \
            /*_mm_prefetch((const char *)(A_data + j + 16), _MM_HINT_T0);*/ \
            /*_mm_prefetch((const char *)(A_j + j + 16), _MM_HINT_T0);*/ \
            CSRMM_FMADD_WO_UNROLL(j); \
            CSRMM_FMADD_WO_UNROLL(j + 1); \
            CSRMM_FMADD_WO_UNROLL(j + 2); \
            CSRMM_FMADD_WO_UNROLL(j + 3); \
          } \
          for (int j = j_begin + len - rem; j < j_end; ++j) { \
            CSRMM_FMADD_WO_UNROLL(j); \
          }

          CSRMM_INNER_PROD_WO_UNROLL;

          for (int n = 0; n < n_rem; ++n) {
            _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, acc[n]);
          }
        }
      } // for each row

      for (A_col_block = 1; A_col_block < num_of_A_col_blocks - 1; ++A_col_block) {
        for (int m = m_begin; m < m_end; ++m) {
          int nn;
          for (nn = 0; nn < n_per_thread/VLEN/CSRMM_REG_BLOCK_SIZE*CSRMM_REG_BLOCK_SIZE; nn += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
            for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
              acc[n] = _MM_LOAD(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN);
            }

            CSRMM_INNER_PROD;

#pragma unroll(CSRMM_REG_BLOCK_SIZE)
            for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
              _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, acc[n]);
            }
          } // for each col block of C

          // remainder col block of C
          if (nn < n_per_thread/VLEN) {
            int n_rem = n_per_thread/VLEN - nn;
            for (int n = 0; n < n_rem; ++n) {
              acc[n] = _MM_LOAD(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN);
            }

            CSRMM_INNER_PROD_WO_UNROLL;

            for (int n = 0; n < n_rem; ++n) {
              _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, acc[n]);
            }
          }
        } // for each row
      } // for each col block of A

      // last col block of A
      for (int m = m_begin; m < m_end; ++m) {
        int nn;
        for (nn = 0; nn < n_per_thread/VLEN/CSRMM_REG_BLOCK_SIZE*CSRMM_REG_BLOCK_SIZE; nn += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
            acc[n] = _MM_LOAD(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN);
          }

          CSRMM_INNER_PROD;

#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
            _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, _MM_MAX(_MM_SETZERO(), acc[n]));
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              float temp[VLEN];
              _MM_STORE(temp, acc[n]);
              printf(" = %g->%g:%d\n",
                  temp[COL_TO_DEBUG%VLEN],
                  C_pr[(m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN],
                  C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN - C);
            }
#endif
          }
        } // for each col block of C

        // remainder col block of C
        if (nn < n_per_thread/VLEN) {
          int n_rem = n_per_thread/VLEN - nn;
          for (int n = 0; n < n_rem; ++n) {
            acc[n] = _MM_LOAD(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN);
          }

          CSRMM_INNER_PROD_WO_UNROLL;

          for (int n = 0; n < n_rem; ++n) {
            _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, _MM_MAX(_MM_SETZERO(), acc[n]));
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              float temp[VLEN];
              _MM_STORE(temp, acc[n]);
              printf(" = %g->%g:%d\n",
                  temp[COL_TO_DEBUG%VLEN],
                  C_pr[(m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN],
                  C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN - C);
            }
#endif
          }
        }
      } // for each row
    } // num_of_A_col_blocks > 1
    else {
      for (int m = m_begin; m < m_end; ++m) {
        int nn;
        for (nn = 0; nn < n_per_thread/VLEN/CSRMM_REG_BLOCK_SIZE*CSRMM_REG_BLOCK_SIZE; nn += CSRMM_REG_BLOCK_SIZE) {
#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
            acc[n] = _MM_SET1(bias[m]);
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              printf("%g ", bias[ROW_TO_DEBUG]);
            }
#endif
            _mm_prefetch((const char *)(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN), _MM_HINT_T0);
          }

          CSRMM_INNER_PROD;

#undef CSRMM_INNER_PROD
#undef CSRMM_FMADD

#pragma unroll(CSRMM_REG_BLOCK_SIZE)
          for (int n = 0; n < CSRMM_REG_BLOCK_SIZE; ++n) {
            _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, _MM_MAX(_MM_SETZERO(), acc[n]));
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              float temp[VLEN];
              _MM_STORE(temp, acc[n]);
              printf(" = %g->%g:%d\n",
                  temp[COL_TO_DEBUG%VLEN],
                  C_pr[(m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN],
                  C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN - C);
            }
#endif
          }
        } // for each col block of C

        // remainder col block of C
        if (nn < n_per_thread/VLEN) {
          int n_rem = n_per_thread/VLEN - nn;
          for (int n = 0; n < n_rem; ++n) {
            acc[n] = _MM_SET1(bias[m]);
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              printf("%g ", bias[ROW_TO_DEBUG]);
            }
#endif
            _mm_prefetch((const char *)(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN), _MM_HINT_T0);
          }

          CSRMM_INNER_PROD_WO_UNROLL;

#undef CSRMM_UNROLL_FACTOR
#undef CSRMM_J_PREFETCH_DISTANCE

          for (int n = 0; n < n_rem; ++n) {
            _MM_STORE(C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN, _MM_MAX(_MM_SETZERO(), acc[n]));
#ifdef DBG_CSRMM
            if (m == ROW_TO_DEBUG && n_begin + (nn + n)*VLEN == COL_TO_DEBUG/VLEN*VLEN) {
              float temp[VLEN];
              _MM_STORE(temp, acc[n]);
              printf(" = %g->%g:%d\n",
                  temp[COL_TO_DEBUG%VLEN],
                  C_pr[(m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN],
                  C_pr + (m - m_begin)*n_per_thread + (nn + n)*VLEN + COL_TO_DEBUG%VLEN - C);
            }
#endif
          }
        }
      } // for each row
    } // num_of_A_col_blocks == 1

    conv_cycles_of_this_batch[tid*16] = __rdtsc() - t;
  } // omp parallel
}

static void __attribute__((noinline)) csrmm(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B,
    float *C,
    int M, int N, int K,
    const float *bias,
    int col_block_size)
{
  int ncolblocks = K/col_block_size;

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
    for (int i = 0; i < M; ++i) {
#ifdef __AVX512F__
      for (int nn = 0; nn < N; nn += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 256
#pragma unroll(16)
        for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
          sum[n] = _mm512_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*M + i]; j < A_i[cb*M + i + 1]; ++j) {
          int c = A_j[j];
          __m512 v_v = _mm512_set1_ps(A_data[j]);
#pragma unroll(16)
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            sum[n] = _mm512_fmadd_ps(v_v, _mm512_load_ps(B + c + nn + n*VLEN), sum[n]);
          }
        }
#pragma unroll(16)
        for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
          _mm512_store_ps(C + i*N + nn + n*VLEN, sum[n]);
        }
      }
#elif defined(__AVX2__)
      for (int nn = 0; nn < N; nn += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 64
        for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
          sum[n] = _mm256_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*M + i]; j < A_i[cb*M + i + 1]; ++j) {
          int c = A_j[j];
          __m256 v_v = _mm256_set1_ps(A_data[j]);
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            sum[n] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + nn + n*VLEN), sum[n]);
          }
        }
        for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
          _mm256_store_ps(C + i*N + nn + n*VLEN, sum[n]);
        }
      }
#else
      for (int nn = 0; nn < N; nn += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 32
        for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
          sum[n] = _mm_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*M + i]; j < A_i[cb*M + i + 1]; ++j) {
          int c = A_j[j];
          __m128 v_v = _mm_set1_ps(A_data[j]);
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            sum[n] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + nn + n*VLEN)), sum[n]);
          }
        }
        for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
          _mm_store_ps(C + i*N + nn + n*VLEN, sum[n]);
        }
      }
#endif
    }

    for (cb = 1; cb < ncolblocks; ++cb) {
#pragma omp for nowait
      for (int i = 0; i < M; ++i) {
#ifdef __AVX512F__
        for (int nn = 0; nn < N; nn += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 64
#pragma unroll(16)
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            sum[n] = _mm512_load_ps(C + i*N + nn + n*VLEN);
          }
          for (int j = A_i[cb*M + i]; j < A_i[cb*M + i + 1]; ++j) {
            int c = A_j[j];
            __m512 v_v = _mm512_set1_ps(A_data[j]);
#pragma unroll(16)
            for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
              sum[n] = _mm512_fmadd_ps(v_v, _mm512_load_ps(B + c + nn + n*VLEN), sum[n]);
            }
          }
#pragma unroll(16)
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            _mm512_store_ps(C + i*N + nn + n*VLEN, sum[n]);
          }
        }
#elif defined(__AVX2__)
        for (int nn = 0; nn < N; nn += REG_BLOCK_SIZE*VLEN) {
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            sum[n] = _mm256_load_ps(C + i*N + nn + n*VLEN);
          }
          for (int j = A_i[cb*M + i]; j < A_i[cb*M + i + 1]; ++j) {
            int c = A_j[j];
            __m256 v_v = _mm256_set1_ps(A_data[j]);
            for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
              sum[n] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + nn + n*VLEN), sum[n]);
            }
          }
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            _mm256_store_ps(C + i*N + nn + n*VLEN, sum[n]);
          }
        }
#else
        for (int nn = 0; nn < N; nn += REG_BLOCK_SIZE*VLEN) { // assume n (batch size) is a multiple of 32
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            sum[n] = _mm_load_ps(C + i*N + nn + n*VLEN);
          }
          for (int j = A_i[cb*M + i]; j < A_i[cb*M + i + 1]; ++j) {
            int c = A_j[j];
            __m128 v_v = _mm_set1_ps(A_data[j]);
            for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
              sum[n] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + nn + n*VLEN)), sum[n]);
            }
          }
          for (int n = 0; n < REG_BLOCK_SIZE; ++n) {
            _mm_store_ps(C + i*N + nn + n*VLEN, sum[n]);
          }
        }
#endif
      }
    } // for each col block
  } // omp parallel
}

#endif /* _CAFFE_UTIL_SPGEMM_HPP_ */
