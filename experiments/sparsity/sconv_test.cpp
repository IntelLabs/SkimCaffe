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

#include "../../include/caffe/util/sconv.hpp"
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
int flop_cnt = 0;

int main(int argc, const char *argv[])
{
  if (argc < 2) {
    fprintf(stderr, "Usage: %s matrix_in_matrix_market_format\n", argv[0]);
    return -1;
  }

#if defined(SNIPER) || defined(SDE)
#ifdef VECTORIZE_OVER_INPUTS
  const int NBATCH = 16;
#else
  const int NBATCH = 1;
#endif
#else
#ifdef VECTORIZE_OVER_INPUTS
  const int NBATCH = 512;
#else
  const int NBATCH = 256;
#endif
#endif

  int nthreads = omp_get_max_threads();
  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups = nthreads/2;
#endif

  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  if (nthread_groups != nthreads) {
    for (int i = 0; i < nthread_groups; ++i) {
      barriers[i] = new synk::Barrier(1, nthreads_per_group);
    }
#pragma omp parallel
    {
      assert(omp_get_num_threads() == nthreads);

      int tid = omp_get_thread_num();
      int gid = tid/nthreads_per_group;
      int tid_in_group = tid%nthreads_per_group;

      barriers[gid]->init(tid_in_group);
    }
  }

  // conv1
//  const int NOUT = 96;
//  const int NIN = 3;
//  const int K = 11;
//  const int WIDTH = 227;
//  const int OUT_WIDTH = 55;
//  const int KERNEL_SIZE_ALIGNED = 128;

  //const float *weight = new float[NOUT * NIN * K * K];
//  const float *weight = new float[NOUT * NIN * KERNEL_SIZE_ALIGNED];
//  const float *input = new float[NBATCH * NIN * WIDTH * WIDTH];
//  float *output = new float[NBATCH * NOUT * OUT_WIDTH * OUT_WIDTH];

  double cpu_freq = SpMP::get_cpu_freq();
  printf("freq = %g\n", cpu_freq);

  // conv3
  const int NOUT = 384;
  const int NIN = 256;
  const int K = 3;
  const int WIDTH = 13;
  const int WOUT = WIDTH;
  const int PAD = 1;
  int colblock = COL_BLOCK;

  SpMP::CSR *A = new SpMP::CSR(argv[1]);
  if (A->m != NOUT) {
    fprintf(stderr, "The input matrix must have %d rows (weight matrix for conv3 in AlexNet)\n", NOUT);
    return -1;
  }
  if (A->n != NIN*K*K) {
    fprintf(stderr, "The input matrix must have %d columns (weight matrix for conv3 in AlexNet)\n", NIN*K*K);
    return -1;
  }
  printf("nnz_proportion = %g\n", (double)A->getNnz()/A->m/A->n);
  float *values = (float *)_mm_malloc(sizeof(float)*A->getNnz(), 4096);

  int ncolblocks = NIN/colblock;

  vector<int *> rowptr_blocked;
  vector<int *> colidx_blocked;
  vector<float *> values_blocked;
  rowptr_blocked.resize(ncolblocks);
  colidx_blocked.resize(ncolblocks);
  values_blocked.resize(ncolblocks);
  std::vector<int> nnzs_of_col_blocks(ncolblocks, 0);

#if defined(SNIPER) || defined(SDE)
  const int REPEAT = 2;
#else
#ifdef VECTORIZE_OVER_INPUTS
  const int REPEAT = 128;
#else
  const int REPEAT = 256;
#endif
#endif

  int *rowptr_split[REPEAT];
  int *colidx_split[REPEAT];
  float *values_split[REPEAT];

  //vector<int *> colidx_interleaved;
  //colidx_interleaved.resize(ncolblocks);

  //int *blockptr_colmajor[REPEAT];
  //int *kidx_colmajor[REPEAT];
  //float *values_colmajor[REPEAT];

  int nnz = A->getNnz();
  int col_major_ic_block = get_col_major_ic_block(nnz, NOUT, NIN);
  printf("col_major_ic_block = %d\n", col_major_ic_block);

  const int SCRATCH_SIZE_PER_IC = WOUT*((WOUT + 16 - 1)/16*16);
  for (int r = 0; r < 1; ++r) {
    //posix_memalign((void **)&blockptr_colmajor[r], 4096, sizeof(int)*(NIN/col_major_ic_block*NOUT + 1));
    //memset(blockptr_colmajor[r], 0, sizeof(int)*(NIN/col_major_ic_block*NOUT + 1));
    //posix_memalign((void **)&kidx_colmajor[r], 4096, sizeof(int)*nnz);
    //posix_memalign((void **)&values_colmajor[r], 4096, sizeof(float)*nnz);

    posix_memalign((void **)&rowptr_split[r], 4096, sizeof(int)*(ncolblocks*NOUT*K + 1));
    memset(rowptr_split[r], 0, sizeof(int)*(ncolblocks*NOUT*K + 1));
    posix_memalign((void **)&colidx_split[r], 4096, sizeof(int)*nnz);
    posix_memalign((void **)&values_split[r], 4096, sizeof(float)*nnz);

    for (int oc = 0; oc < NOUT; ++oc) {
      for (int j = A->rowptr[oc]; j < A->rowptr[oc + 1]; ++j) {
        int col = A->colidx[j];

        int kernel_col = col%K;
        int kernel_row = (col/K)%K;
        int ic = col/(K*K);
        assert(ic < NIN);

        A->colidx[j] = (ic*(WIDTH + PAD) + kernel_row)*(WIDTH + PAD) + kernel_col;
        values[j] = A->values[j];

        int bcol = ic/colblock;
        ++nnzs_of_col_blocks[bcol];

        ++rowptr_split[r][(ic/colblock*NOUT + oc)*K + K - 1 - kernel_col + 1];

        int bcol_colmajor = ic/col_major_ic_block;
        //++blockptr_colmajor[r][bcol_colmajor*NOUT + oc + 1];
      }
    }

    for (int i = 0; i < ncolblocks; ++i) {
  //    rowptr_blocked[i] = (int *)malloc_huge_pages(sizeof(int)*(NOUT + 1));
  //    colidx_blocked[i] = (int *)malloc_huge_pages(sizeof(int)*nnzs_of_col_blocks[i]);
  //    values_blocked[i] = (float *)malloc_huge_pages(sizeof(float)*nnzs_of_col_blocks[i]);

      posix_memalign((void **)&rowptr_blocked[i], 4096, sizeof(int)*(NOUT + 1));
      posix_memalign((void **)&colidx_blocked[i], 4096, sizeof(int)*nnzs_of_col_blocks[i]);
      posix_memalign((void **)&values_blocked[i], 4096, sizeof(float)*nnzs_of_col_blocks[i]);
      //posix_memalign((void **)&colidx_interleaved[i], 4096, sizeof(float)*nnzs_of_col_blocks[i]);
      nnzs_of_col_blocks[i] = 0;
      rowptr_blocked[i][0] = 0;
    }

    //for (int i = 1; i < NIN/col_major_ic_block*NOUT; ++i) {
      //blockptr_colmajor[r][i + 1] += blockptr_colmajor[r][i];
    //}
    for (int i = 1; i < ncolblocks*NOUT*K; ++i) {
      rowptr_split[r][i + 1] += rowptr_split[r][i];
    }
    assert(rowptr_split[r][ncolblocks*NOUT*K] == nnz);

    for (int oc = 0; oc < NOUT; ++oc) {
      for (int j = A->rowptr[oc]; j < A->rowptr[oc + 1]; ++j) {
        int c = A->colidx[j];
        int kernel_col = c%(WIDTH + PAD);
        int kernel_row = c/(WIDTH + PAD)%(WIDTH + PAD);
        int ic = c/(WIDTH + PAD)/(WIDTH + PAD);
        int bcol = ic/colblock;

        colidx_blocked[bcol][nnzs_of_col_blocks[bcol]] = c;
        values_blocked[bcol][nnzs_of_col_blocks[bcol]] = values[j];
        //colidx_interleaved[bcol][nnzs_of_col_blocks[bcol]] = c*VLEN;
        nnzs_of_col_blocks[bcol]++;

        int splitid = (ic/colblock*NOUT + oc)*K + (K - 1 - kernel_col);
        colidx_split[r][rowptr_split[r][splitid]] = (ic*(WIDTH + PAD) + kernel_row)*16;
        values_split[r][rowptr_split[r][splitid]] = values[j];
        ++rowptr_split[r][splitid];

        //int blockid = ic/col_major_ic_block*NOUT + oc;
        //int offset = blockptr_colmajor[r][blockid];
        //kidx_colmajor[r][offset] = ((ic%col_major_ic_block*K + kernel_col)*(WOUT + PAD) + kernel_row)*16;
        //values_colmajor[r][offset] = values[j];
        //++blockptr_colmajor[r][blockid];
      }

      for (int i = 0; i < ncolblocks; ++i) {
        rowptr_blocked[i][oc + 1] = nnzs_of_col_blocks[i];
      }
    }

    //for (int i = NIN/col_major_ic_block*NOUT - 1; i > 0; --i) {
      //blockptr_colmajor[r][i] = blockptr_colmajor[r][i - 1];
    //}
    //blockptr_colmajor[r][0] = 0;
    for (int i = ncolblocks*NOUT*K - 1; i > 0; --i) {
      rowptr_split[r][i] = rowptr_split[r][i - 1];
    }
    rowptr_split[r][0] = 0;
    //for (int out_channel = 0; out_channel < NOUT; ++out_channel) {
      //int nnz_of_oc = 0;
      //for (int i = 0; i < NIN/col_major_ic_block; ++i) {
        //nnz_of_oc += blockptr_colmajor[r][i*NOUT + out_channel + 1] - blockptr_colmajor[r][i*NOUT + out_channel];
      //}
      //if (nnz_of_oc != A->rowptr[out_channel + 1] - A->rowptr[out_channel]) {
        //printf("oc = %d rowptr[oc+1] - rowptr[oc] expected %d actual %d\n", out_channel, A->rowptr[out_channel + 1] - A->rowptr[out_channel], nnz_of_oc);
      //}
    //}
  }

  size_t input_size = sizeof(float)*NBATCH*(NIN*(WIDTH + PAD)*(WIDTH + PAD) + PAD*(WIDTH + 2*PAD));
  float *input = (float *)_mm_malloc(input_size, 4096);
//  float *input = (float *)malloc_huge_pages(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
#pragma omp parallel for
  for (int i = 0; i < input_size/sizeof(float); ++i) {
    input[i] = i;
  }

  size_t output_size = sizeof(float)*1*NBATCH*NOUT*WOUT*WOUT;
  float *output = (float *)_mm_malloc(output_size, 4096);
#pragma omp parallel for
  for (int i = 0; i < output_size/sizeof(float); ++i) {
    output[i] = 0;
  }

  float *bias = (float *)_mm_malloc(sizeof(float)*NOUT, 4096);
//  float *bias = (float *)malloc_huge_pages(sizeof(float)*NOUT);
  for (int i = 0; i < NOUT; ++i) {
    bias[i] = -i;
  }
  size_t scratch_size = sizeof(float)*OC_BLOCK*WOUT*((WOUT + 16 - 1)/16*16)*nthreads;
  float *scratch = (float *)_mm_malloc(scratch_size, 4096);
  memset((void *)scratch, 0, scratch_size);

  /*float *input_scratch = (float *)_mm_malloc(sizeof(float)*nthreads*col_major_ic_block*K*K*SCRATCH_SIZE_PER_IC, 4096);
  memset((void *)input_scratch, 0, sizeof(float)*nthreads*col_major_ic_block*K*K*SCRATCH_SIZE_PER_IC);*/

  float *output_colmajor_scratch;
#ifdef COL_MAJOR_OC_BLOCK
  posix_memalign((void **)&output_colmajor_scratch, 4096, sizeof(float)*nthreads*COL_MAJOR_OC_BLOCK*SCRATCH_SIZE_PER_IC);
#else
  posix_memalign((void **)&output_colmajor_scratch, 4096, sizeof(float)*nthreads*NOUT*SCRATCH_SIZE_PER_IC);
#endif
  memset((void *)output_colmajor_scratch, 0, sizeof(float)*nthreads*NOUT*SCRATCH_SIZE_PER_IC);

  unsigned long long times[nthreads*16];
  for (int tid = 0; tid < nthreads; ++tid) {
    times[tid*16] = 0;
    conv_cycles_of_this_batch[tid*16] = 0;
  }

  printf("REPEAT = %d, NBATCH = %d\n", REPEAT, NBATCH);

  const int **rowptr_blocked_temp = (const int **)&rowptr_blocked[0];
  const int **colidx_blocked_temp = (const int **)&colidx_blocked[0];
  const float **values_blocked_temp = (const float **)&values_blocked[0];
  //const int **colidx_interleaved_temp = (const int **)&colidx_interleaved[0];

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

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    for (int j = 0; j < REPEAT; ++j) {
      if (j == REPEAT - 1 && 0 == tid) {
#ifdef SNIPER
        SimRoiStart();
#endif
#ifdef SDE
        ssc_start_performance();
#endif
      }

      int nthread_groups = nthreads;
#ifdef __AVX512F__
      nthread_groups = NTILES;
#endif
      assert(nthreads%nthread_groups == 0);
      int nthreads_per_group = nthreads/nthread_groups;
      int gid = tid/nthreads_per_group;

#ifdef VECTORIZE_OVER_INPUTS
      int i_per_group = (NBATCH/VLEN + nthread_groups - 1)/nthread_groups;
      int i_begin = std::min(i_per_group*gid, NBATCH/VLEN);
      int i_end = std::min(i_begin + i_per_group, NBATCH/VLEN);
      i_begin *= VLEN;
      i_end *= VLEN;
//      printf("[%d] %d-%d\n", tid, i_begin, i_end);
#else
      int i_per_group = (NBATCH + nthread_groups - 1)/nthread_groups;
      int i_begin = std::min(i_per_group*gid, NBATCH);
      int i_end = std::min(i_begin + i_per_group, NBATCH);
#endif

      unsigned long long tt = __rdtsc();

      for (int i = i_begin; i < i_end; ++i) {
#ifdef VECTORIZE_OVER_INPUTS
        if (i%VLEN == 0) {
          sconv345_vectorize_over_inputs(
              input + (j*NBATCH + i)*NIN*(WIDTH + PAD)*(WIDTH + PAD),
              rowptr_blocked_temp, colidx_interleaved_temp, values_blocked_temp,
              ncolblocks,
              bias,
              output + (j*NBATCH + i)*NOUT*WOUT*WOUT,
              NOUT, NIN);
        }
#else
//        sconv345_ver2(
//            input + i*NIN*(WIDTH + PAD)*(WIDTH + PAD),
//            input + i*NIN*(WIDTH + PAD)*(WIDTH + PAD),
//            NIN,
//            blockptr_colmajor, kidx_colmajor, values_colmajor,
//            bias,
//            output + i*NOUT*WOUT*WOUT, NOUT,
//            input_scratch, output_colmajor_scratch, col_major_ic_block);
        sconv_3x3_pad1<13>(
              input + i*NIN*(WIDTH + PAD)*(WIDTH + PAD),
              //A->rowptr, A->colidx, values,
              rowptr_blocked_temp, colidx_blocked_temp, values_blocked_temp,
              ncolblocks,
              bias,
              output + i*NOUT*WOUT*WOUT, NOUT,
              scratch + tid*OC_BLOCK*WOUT*((WOUT + 16 - 1)/16*16));
          //sconv345_split(
              //input + (/*j*NBATCH*/ + i)*NIN*(WIDTH + PAD)*16,
              //rowptr_split[0], colidx_split[0], values_split[0],
              //ncolblocks,
              //bias,
              //output + (/*j*NBATCH*/ + i)*NOUT*(WIDTH + PAD)*(WIDTH + PAD), NOUT,
              //scratch + tid*OC_BLOCK*WIDTH*16);
#endif
      }

      times[tid*16] += __rdtsc() - tt;

      if (j == REPEAT - 1 && 0 == tid) {
#ifdef SNIPER
        SimRoiEnd();
#endif
#ifdef SDE
        ssc_stop_performance();
#endif
      }
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

  // correctness check
  float *output_ref = (float *)_mm_malloc(output_size, 4096);

#pragma omp parallel for
  for (int i = 0; i < NBATCH; ++i) {
    caffe_cpu_sconv_default(
        input + i*NIN*(WIDTH + PAD)*(WIDTH + PAD), NIN,
        WIDTH, WIDTH,
        PAD, PAD,
        1, 1,
        1, 1,
        A->rowptr, A->colidx, values,
        K, K,
        bias,
        output_ref + i*NOUT*WOUT*WOUT, NOUT);
  }

  for (int i = 0; i < NBATCH; ++i) {
    for (int j = 0; j < NOUT; ++j) {
      for (int k = 0; k < WOUT; ++k) {
        for (int l = 0; l < WOUT; ++l) {
          float expected = output_ref[((i*NOUT + j)*WOUT + k)*WOUT + l];
          float actual = output[((i*NOUT + j)*WOUT + k)*WOUT + l];
          if (fabs(expected - actual)/fabs(expected) > 1e-1) {
            printf("(%d, %d, %d, %d) expected %g actual %g\n", i, j, k, l, expected, actual);
            return -1;
          }
        }
      }
    }
  }

  unsigned long long max_cycles = 0, max_cycles2 = 0;
  unsigned long long sum_cycles = 0;
  for (int i = 0; i < nthreads; ++i) {
    max_cycles = std::max(max_cycles, times[i*16]);
    max_cycles2 = std::max(max_cycles, conv_cycles_of_this_batch[i*16]);
    sum_cycles += times[i*16];
  }

  double flops = (double)NOUT*NIN*WIDTH*WIDTH*K*K*2;
  //printf("flops = %lld\n", flop_cnt);
  printf("mflops-per-file %g\n", flops/1e6);
  printf("effective-GF/s %g %g\n", flops*REPEAT*NBATCH/t/1e9, flops*REPEAT*NBATCH/(max_cycles/cpu_freq)/1e6);
  printf("wall_clock_time = %g, max_time = %g, avg_time = %g, tt = %g\n", t, max_cycles/cpu_freq, (double)sum_cycles/omp_get_max_threads()/cpu_freq, max_cycles2/cpu_freq);

  return 0;
}
