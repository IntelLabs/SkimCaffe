/*
 * conv1_test.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: jpark103
 */

#include <cstdio>
#include <omp.h>
#include <immintrin.h>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>

#include "../../include/caffe/util/conv.hpp"
#include "../../include/caffe/util/spgemm.hpp"
#include "SpMP/CSR.hpp"
#include <mkl.h>

#ifdef SEP
#include "sampling.h"
#endif

#ifdef VTUNE
#include "ittnotify.h"
#endif

#ifdef SNIPER
#include "sim_api.h"
#endif

#ifdef SDE
#include "../../include/caffe/util/sde.h"
#endif

using namespace std;

synk::Barrier *barriers[256];

unsigned long long conv_cycles_of_this_batch[1024*16];
unsigned long long reduce_cycles[1024*16];
int flop_cnt = 0;

int main(int argc, const char *argv[])
{
#define FC

  const int NBATCH = 256;

  int nthreads = omp_get_max_threads();

  double cpu_freq = SpMP::get_cpu_freq();
  printf("freq = %g\n", cpu_freq);

  SpMP::CSR *A = new SpMP::CSR(argv[1]);
#if defined(SNIPER) || defined(SDE)
  int NOUT = A->m/32; // scale down to 1 tile
#elif defined(SDE)
  int NOUT = A->m/64; // scale down to 1 core
#else
  int NOUT = A->m;
#endif
  int NIN = A->n;

//#define B_DECOMPOSITION

#ifdef B_DECOMPOSITION
  typedef int idx_t;
#ifdef __AVX512F__
  int num_of_B_col_partitions = NBATCH/(4*VLEN);
#else
  int num_of_B_col_partitions = NBATCH/(8*VLEN);
#endif
  assert(nthreads%num_of_B_col_partitions == 0);
  int num_of_B_row_partitions = nthreads/num_of_B_col_partitions;
    // output matrix C will be replicated by num_of_B_row_blocks times
  printf(
      "num_of_B_row_partitions = %d, num_of_B_col_partitions = %d\n",
      num_of_B_row_partitions, num_of_B_col_partitions);
  int num_of_A_col_blocks = num_of_B_row_partitions;
  int num_of_A_row_blocks = 1;

  for (int i = 0; i < num_of_B_col_partitions; ++i) {
    barriers[i] = new synk::Barrier(num_of_B_row_partitions, 1);
  }
#pragma omp parallel
  {
    assert(omp_get_num_threads() == nthreads);

    int tid = omp_get_thread_num();
    int gid = tid/num_of_B_row_partitions;
    int tid_in_group = tid%num_of_B_row_partitions;

    barriers[gid]->init(tid_in_group);
  }
#else
  // C decomposition
  typedef int idx_t;

  int A_col_block_size = argc > 2 ? atoi(argv[2]) : 256;
  int num_of_A_col_blocks = NIN/A_col_block_size;

  int num_of_C_col_partitions = NBATCH/(VLEN*CSRMM_REG_BLOCK_SIZE);
    // AVX512: 256/(16*4) = 4, AVX2: 256/(8*8) = 4, SSE: 64/(4*8) = 2
  if (nthreads%num_of_C_col_partitions != 0) {
    fprintf(stderr, "num_of_C_col_partitions %d should divide # of threads %d\n", num_of_C_col_partitions, nthreads);
    return -1;
  }
  int num_of_C_row_partitions = nthreads/num_of_C_col_partitions;

  int num_of_A_row_blocks = num_of_C_row_partitions;
  printf("num_of_A_col_blocks = %d, num_of_C_row_partitions = %d, num_of_C_col_partitions = %d\n", num_of_A_col_blocks, num_of_C_row_partitions, num_of_C_col_partitions);
#endif

  int nnz = A->getNnz();

  // 2D blocking of weight matrix A
  int *weight_i_blocked_;
  idx_t *weight_j_blocked_;
  float *weight_values_blocked_;

  posix_memalign((void **)&weight_i_blocked_, 4096, sizeof(int)*(NOUT*num_of_A_col_blocks + 1));
  posix_memalign((void **)&weight_j_blocked_, 4096, sizeof(idx_t)*nnz);
  posix_memalign((void **)&weight_values_blocked_, 4096, sizeof(float)*nnz);

  weight_i_blocked_[0] = 0;
  nnz = 0;
  int i_per_row_block = (NOUT + num_of_A_row_blocks - 1)/num_of_A_row_blocks;
  int j_per_col_block = (NIN + num_of_A_col_blocks - 1)/num_of_A_col_blocks;

  for (int row_block = 0; row_block < num_of_A_row_blocks; ++row_block) {
    int i_begin = std::min(i_per_row_block*row_block, NOUT);
    int i_end = std::min(i_begin + i_per_row_block, NOUT);

    for (int col_block = 0; col_block < num_of_A_col_blocks; ++col_block) {
      int c_begin = std::min(j_per_col_block*col_block, NIN);
      int c_end = std::min(c_begin + j_per_col_block, NIN);

      for (int i = i_begin; i < i_end; ++i) {
        for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
          int c = A->colidx[j];
          if (c >= c_begin && c < c_end) {
            if (sizeof(idx_t) == 2) {
              weight_j_blocked_[nnz] = c;
            }
            else {
              // When we have enough bits for column indices,
              // we pre-multiply it with # of columns of matrix B
              weight_j_blocked_[nnz] = NBATCH/num_of_C_col_partitions*c;
            }
            weight_values_blocked_[nnz] = A->values[j];
            ++nnz;
          }
        }
        weight_i_blocked_[num_of_A_col_blocks*i_begin + col_block*(i_end - i_begin) + (i - i_begin) + 1] = nnz;
      }
    } // for each col block
  } // for each row block
  assert(nnz == A->getNnz());

  size_t input_size = sizeof(float)*NBATCH*NIN;
  float *input = (float *)_mm_malloc(input_size, 4096);
//  float *input = (float *)malloc_huge_pages(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
  for (int i = 0; i < input_size/sizeof(float); ++i) {
    input[i] = i%123;
  }

  // rearrange input
  float *input_rearranged = (float *)_mm_malloc(input_size, 4096);
  int col_block_size = NBATCH/num_of_C_col_partitions;
  for (int col_block = 0; col_block < num_of_C_col_partitions; ++col_block) {
    for (int i = 0; i < NIN; ++i) {
      for (int j = 0; j < col_block_size; ++j) {
        input_rearranged[(col_block*NIN + i)*col_block_size + j] =
            input[i*NBATCH + col_block*col_block_size + j];
      }
    }
  }

  size_t output_size = sizeof(float)*NOUT*NIN;
  float *output = (float *)_mm_malloc(output_size, 4096);
//  float *output = (float *)malloc_huge_pages(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
  memset((void *)output, 0, output_size);

  float *output_scratch = (float *)_mm_malloc(output_size*nthreads, 4096);
  memset((void *)output_scratch, 0, output_size*nthreads);

  float *bias = (float *)_mm_malloc(sizeof(float)*NOUT, 4096);
//  float *bias = (float *)malloc_huge_pages(sizeof(float)*NOUT);
  for (int i = 0; i < NOUT; ++i) {
    bias[i] = -(i%123);
  }

//  printf(
//      "input = %p, weight_j_blocked_ = %p, weight_values_blocked_ = %p, output = %p, weight_i_blocked = %p\n",
//      input, weight_j_blocked_, weight_values_blocked_, output, weight_i_blocked_);

  unsigned long long times[nthreads*16];
  for (int tid = 0; tid < nthreads; ++tid) {
    times[tid*16] = 0;
    conv_cycles_of_this_batch[tid*16] = 0;
    reduce_cycles[tid*16] = 0;
  }

#if defined(SNIPER) || defined(SDE)
  const int REPEAT = 2;
#else
#ifdef VECTORIZE_OVER_INPUTS
  const int REPEAT = 128;
#else
  const int REPEAT = 256;
#endif
#endif

  printf("REPEAT = %d, NBATCH = %d\n", REPEAT, NBATCH);

#ifdef SEP
  VTResumeSampling();
#endif
#ifdef VTUNE
  fprintf(stderr, "__itt_resume\n");
  __itt_resume();
#endif
#ifdef SNIPER
  SimWarmup();
#endif
#ifdef SDE
  ssc_initialization();
#endif

  double t = omp_get_wtime();

  for (int j = 0; j < REPEAT; ++j) {
    if (j == REPEAT - 1) {
#ifdef SNIPER
      SimRoiStart();
#endif
#ifdef SDE
      ssc_start_performance();
#endif
    }

#ifdef B_DECOMPOSITION
    // 2D decomposition of B
    csrmm_fused_B_decomposed(
        weight_values_blocked_, weight_j_blocked_, weight_i_blocked_,
        input,
        output,
        NOUT, NBATCH, NIN,
        bias,
        output_scratch,
        num_of_B_row_partitions, num_of_B_col_partitions);
#else
    // 2D decomposition of C
    csrmm_fused_C_decomposed(
        weight_values_blocked_, weight_j_blocked_, weight_i_blocked_,
        input_rearranged,
        output,
        NOUT, NBATCH, NIN,
        bias,
        num_of_C_row_partitions, num_of_C_col_partitions,
        num_of_A_col_blocks);
#endif

    if (j == REPEAT - 1) {
#ifdef SNIPER
      SimRoiEnd();
#endif
#ifdef SDE
      ssc_stop_performance();
#endif
    }
  }

#ifdef SEP
  VTPauseSampling();
#endif
#ifdef VTUNE
  __itt_pause();
  fprintf(stderr, "__itt_pause\n");
#endif
#ifdef SDE
  ssc_stop_simulation();
#endif

  t = omp_get_wtime() - t;

  // Re-arrange output matrix C
  // In csrmm_fused_C_decomposed, each thread writes to the contiguous locations for spatial locality
  // which is not necessarily match with the original layout of output matarix C.
  float *temp_output = new float[NOUT*NIN];
  int i_per_block = (NOUT + num_of_C_row_partitions - 1)/num_of_C_row_partitions;
  int j_per_block = (NBATCH + num_of_C_col_partitions - 1)/num_of_C_col_partitions;
#pragma omp parallel for
  for (int row_block = 0; row_block < num_of_C_row_partitions; ++row_block) {
    int i_begin = std::min(i_per_block*row_block, NOUT);
    int i_end = std::min(i_begin + i_per_block, NOUT);

    for (int col_block = 0; col_block < num_of_C_col_partitions; ++col_block) {
      int j_begin = std::min(j_per_block*col_block, NBATCH);
      int j_end = std::min(j_begin + j_per_block, NBATCH);

      for (int i = i_begin; i < i_end; ++i) {
        for (int j = j_begin; j < j_end; ++j) {
          temp_output[i*NBATCH + j] =
              output[((row_block*num_of_C_col_partitions + col_block)*i_per_block + i - i_begin)*j_per_block + j - j_begin];
        }
      }
    }
  }
  memcpy(output, temp_output, sizeof(float)*NOUT*NIN);
  delete[] temp_output;

  unsigned long long max_spmdm_cycles = 0, sum_spmdm_cycles = 0;
  unsigned long long max_reduce_cycles = 0, sum_reduce_cycles = 0;
  for (int tid = 0; tid < nthreads; ++tid) {
    max_spmdm_cycles = std::max(max_spmdm_cycles, conv_cycles_of_this_batch[tid*16]);
    sum_spmdm_cycles += conv_cycles_of_this_batch[tid*16];

    max_reduce_cycles = std::max(max_reduce_cycles, reduce_cycles[tid*16]);
    sum_reduce_cycles += reduce_cycles[tid*16];
  }

  // correctness check
  const char *matdescra = "GXXCX";
  const char transa = 'N';
  float alpha = 1;
  float beta = 0;
  float *temp_values = new float[A->getNnz()];
  for (int i = 0; i < A->getNnz(); ++i) {
    temp_values[i] = A->values[i];
  }
  float *output_ref = new float[output_size/sizeof(float)];
  mkl_scsrmm(&transa, &NOUT, &NBATCH, &NIN, &alpha, matdescra, temp_values, A->colidx, A->rowptr, A->rowptr + 1, input, &NBATCH, &beta, output_ref, &NBATCH);
  for (int i = 0; i < NOUT; ++i) {
    for (int j = 0; j < NBATCH; ++j) {
      output_ref[i*NBATCH + j] = std::max<float>(output_ref[i*NBATCH + j] + bias[i], 0);
    }
  }

#ifdef DBG_CSRMM
  printf("%g ", bias[ROW_TO_DEBUG]);
  float sum = bias[ROW_TO_DEBUG];
  for (int j = A->rowptr[ROW_TO_DEBUG]; j < A->rowptr[ROW_TO_DEBUG + 1]; ++j) {
    float w = temp_values[j];
    int off = A->colidx[j];
    printf(" + %g*%d:%g ", w, off, input[off*NBATCH + COL_TO_DEBUG]);
    sum += w*input[off*NBATCH + COL_TO_DEBUG];
  }
  printf(" = %g\n", sum);
#endif

  for (int i = 0; i < NOUT; ++i) {
    for (int j = 0; j < NBATCH; ++j) {
      if (fabs(output_ref[i*NBATCH + j] - output[i*NBATCH + j])/fabs(output_ref[i*NBATCH + j]) > 1e-1) {
        printf("(%d, %d) expected %g actual %g\n", i, j, output_ref[i*NBATCH + j], output[i*NBATCH + j]);
        return -1;
      }
    }
  }

  delete[] temp_values;
  delete[] output_ref;

  double flops = (double)NOUT*NIN*2;
  printf("mflops-per-file %g\n", flops/1e6);
  printf("effective-GF/s %g %g\n", flops*REPEAT*NBATCH/t/1e9, flops*NBATCH/(max_spmdm_cycles/cpu_freq)/1e6);
  printf("wall_clock_time = %g, max_spmdm_time = %g, avg_spmdm_time = %g, max_reduce_time = %g, avx_reduce_time = %g\n", t/REPEAT, max_spmdm_cycles/cpu_freq, (double)sum_spmdm_cycles/nthreads/cpu_freq, max_reduce_cycles/cpu_freq, (double)sum_reduce_cycles/nthreads/cpu_freq);

  return 0;
}
