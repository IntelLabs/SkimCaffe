/*
 * conv.hpp
 *
 *  Created on: Apr 17, 2016
 *      Author: jpark103
 */

#ifndef _CAFFE_UTIL_CONV_HPP_
#define _CAFFE_UTIL_CONV_HPP_

#include <vector>
#include <immintrin.h>
#include "SpMP/synk/barrier.hpp"
#include "intrinsic.hpp"

#ifdef __AVX512F__
#ifdef SNIPER
static const int NTILES = 1; // 1 tile
#else
static const int NTILES = 64; // FIXME - hardcoded for 68c KNL
#endif
#endif

static const int OC_BLOCK = 16;
static const int COL_BLOCK = 32;

//#define VECTORIZE_OVER_INPUTS

//static const int COL_MAJOR_IC_BLOCK = 8;
//static const int COL_MAJOR_OC_BLOCK = 64;

extern synk::Barrier *barriers[256];

extern unsigned long long conv_cycles_of_this_batch[1024*16], transpose_cycle, pool_cycle;

static int get_col_major_ic_block(int nnz, int num_out_channels, int num_in_channels) {
  // # of in-channels to have on average 8 non-zeros per out-channel
  double nnz_per_oc_and_ic = (double)nnz/num_out_channels/num_in_channels;
  int ret = std::max(8, 1 << (int)round(log2(std::max(1., 8/nnz_per_oc_and_ic))));
  while (num_in_channels%ret != 0) {
    ++ret;
  }
  return ret;
}

extern int flop_cnt;

/**
 * Direct sparse convolution optimized for 3-5 layers of AlexNet
 *
 * This version involves a lot of unaligned loads
// JSP: AlexNet each group of conv3-5
// Input: 256 x 15 x 15 => 900 B per channel, 225 KB total
// Output: 384 x 13 x 13 => 676 B per channel, 253 KB total
// Weight: 384 x 256 x 3 x 3 => 72B per channel pair, 18 KB per output channel, 27 KB per input channel, 6.8 MB total
//         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
// FLOPS: 2 x 384 x 256 x 13 x 13 x 3 x 3 = 299 MFLOPS


 */
template<int WIDTH>
static /*inline*/ void __attribute__((noinline)) sconv_3x3_pad1(
    // input features
    const float *input,
    // weights
    const int **rowptr_blocked, const int **colidx_blocked, const float **values_blocked,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int out_channels,
    float *scratch) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WOUT = WIDTH;
  const int PAD = 1;

  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups /= 2; // 1 group per core in Xeon
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  int c_per_thread = (out_channels/OC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
  int c_begin = std::min(c_per_thread*tid_in_group, out_channels/OC_BLOCK);
  int c_end = std::min(c_begin + c_per_thread, out_channels/OC_BLOCK);
  const int ALIGNED_W = (WIDTH + 16 - 1)/16*16;

#if !defined(__AVX512F__) && defined(__AVX2__)
  const int REG_BLOCK_SIZE = 14; // use at most 14 SIMD registers
  const int REG_BLOCK_W = (WIDTH + VLEN - 1)/VLEN;
  const int REG_BLOCK_H = REG_BLOCK_SIZE/REG_BLOCK_W;
  // WIDTH = 13 (AlexNet conv3-5), AVX2 : REG_BLOCK_W = 2, REG_BLOCK_H = 7, ALIGNED_W = 16
  // WIDTH = 56 (GoogLeNet), AVX2 : REG_BLOCK_W = 7, REG_BLOCK_H = 2, ALIGNED_W = 64

  __declspec(aligned(64)) int mask_temp[VLEN] = { 0 };
  for (int i = 0; i < WIDTH%VLEN; ++i) {
    mask_temp[i] = -1;
  }
  __m256i mask_v = _mm256_load_si256((__m256i *)mask_temp);
#endif

  for (int oc_begin = c_begin*OC_BLOCK; oc_begin < c_end*OC_BLOCK; oc_begin += OC_BLOCK) {
#ifdef __AVX512F__
    __m512 sum[WOUT];

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    int hbegin = 0, hend = WOUT;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
      __m512 bias_v = _mm512_set1_ps(bias[oc]);

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = bias_v;
      }

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

#define W_PREFETCH_DISTANCE (1)

//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
      }
    } // for each oc channel

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      hbegin = 0, hend = 13;
      for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
        int jbegin = rowptr[oc];
        int jend = rowptr[oc + 1];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
          //_mm_prefetch((const char *)(scratch + ((oc - oc_begin + 1)*WOUT + h)*16), _MM_HINT_T0);
          _mm_prefetch((const char *)(values + jend + (h - hbegin)*16), _MM_HINT_T0);
          _mm_prefetch((const char *)(colidx + jend + (h - hbegin)*16), _MM_HINT_T0);
        }

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(13)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin]);
          }
        }

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
        }
      } // for each out channel
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    hbegin = 0; hend = 13;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
        _mm_prefetch((const char *)(output + oc*WOUT*WOUT + h*16), _MM_HINT_T0);
      }

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

//      _mm_prefetch((const char *)(values + rowptr[oc + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(values + rowptr[oc + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[oc + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[oc + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_mask_storeu_ps(output + (oc*WOUT + h)*WOUT, 0x1fff, sum[h - hbegin]);
      }
    }
#elif defined(__AVX2__)
    __m256 sum[REG_BLOCK_H][REG_BLOCK_W];
    __m256 w_v;
    int off;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {
      __m256 bias_v = _mm256_set1_ps(bias[out_channel]);

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      // register blocking over input image positions
      int hbegin;
      for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
        int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H) // compiler gives warning for unroll pragma, but it still unrolls as we want.
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            sum[h - hbegin][w] = bias_v;

#ifdef DBG_SCONV
            if (out_channel == 359 && h == 32 && 28 >= w*VLEN && 28 < (w + 1)*VLEN) {
              float temp[VLEN];
              _mm256_store_ps(temp, bias_v);
              printf("%g", temp[28 - w*VLEN]);
            }
#endif
          }
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(REG_BLOCK_H)
          for (int h = 0; h < REG_BLOCK_H; ++h) { // by some reason, iterating from hbegin to hend prevents icc from unrolling
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h][w] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + (h + hbegin)*(WIDTH + PAD) + VLEN*w), sum[h][w]);
            }

#ifdef DBG_SCONV
            if (out_channel == 359 && h == 32) {
              float temp[VLEN];
              _mm256_store_ps(temp, sum[h - hbegin][28/VLEN]);
              printf(" + %g*%d:%g:%g", values[j], off, input[off + 32*(WIDTH + PAD) + 28], temp[28%VLEN]);
            }
#endif
          }
        }

#pragma unroll(REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
          }
        }
      } // for each register block

      // remainder register block
      if (WOUT%REG_BLOCK_H != 0) {
        // Lower half of images
        int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            sum[h - hbegin][w] = bias_v;
          }
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + VLEN*w), sum[h - hbegin][w]);
            }
          }
        }

#pragma unroll(WOUT%REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
          }
        }
      } // remainder register block
    } // for each output channel

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {
        int jbegin = rowptr[out_channel];
        int jend = rowptr[out_channel + 1];

        // register blocking over input image positions
        int hbegin;
        for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
          int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
            }
          }

          for (int j = jbegin; j < jend; ++j) {
            w_v = _mm256_set1_ps(values[j]);
            off = colidx[j];

#pragma unroll(REG_BLOCK_H)
            for (int h = 0; h < REG_BLOCK_H; ++h) {
#pragma unroll(REG_BLOCK_W)
              for (int w = 0; w < REG_BLOCK_W; ++w) {
                sum[h][w] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + (h + hbegin)*(WIDTH + PAD) + VLEN*w), sum[h][w]);
              }
#ifdef DBG_SCONV
              if (out_channel == 359 && h == 32) {
                float temp[VLEN];
                _mm256_store_ps(temp, sum[h - hbegin][28/VLEN]);
                printf(" + %g*%d:%g:%g", values[j], off, input[off + 32*(WIDTH + PAD) + 28], temp[28%VLEN]);
              }
#endif
            }
          }

#pragma unroll(REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
            }
          }
        } // for each register block

        // remainder register block
        if (WOUT%REG_BLOCK_H != 0) {
          int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
            }
          }

          for (int j = jbegin; j < jend; ++j) {
            w_v = _mm256_set1_ps(values[j]);
            off = colidx[j];

#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
              for (int w = 0; w < REG_BLOCK_W; ++w) {
                sum[h - hbegin][w] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + VLEN*w), sum[h - hbegin][w]);
              }
            }
          }

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
            }
          }
        } // remainder register block
      } // for each output channel
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {
      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      // register blocking over input image positions
      int hbegin;
      for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
        int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            sum[h - hbegin][w] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
          }
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(REG_BLOCK_H)
          for (int h = 0; h < REG_BLOCK_H; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h][w] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + (h + hbegin)*(WIDTH + PAD) + VLEN*w), sum[h][w]);
            }

#ifdef DBG_SCONV
            if (out_channel == 359 && h == 32) {
              float temp[VLEN];
              _mm256_store_ps(temp, sum[h - hbegin][28/VLEN]);
              printf(" + %g*%d:%g:%g", values[j], off, input[off + 32*(WIDTH + PAD) + 28], temp[28%VLEN]);
            }
#endif
          }
        }

#pragma unroll(REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
          if (WIDTH%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }

#ifdef DBG_SCONV
            if (out_channel == 359 && h == 32) {
              printf(" = %g\n", output[(out_channel*WOUT + h)*WOUT + 28]);
            }
#endif
          }
          else {
            int w;
#pragma unroll(REG_BLOCK_W - 1)
            for (w = 0; w < REG_BLOCK_W - 1; ++w) {
              _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }
            _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
          }
        }
      } // remainder register block

      // remainder register block
      if (WOUT%REG_BLOCK_H != 0) {
        int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            sum[h - hbegin][w] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
          }
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + VLEN*w), sum[h - hbegin][w]);
            }
          }
        }

#pragma unroll(WOUT%REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
          if (WIDTH%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }
          }
          else {
            int w;
#pragma unroll(REG_BLOCK_W - 1)
            for (w = 0; w < REG_BLOCK_W - 1; ++w) {
              _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }
            _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
          }
        }
      } // remainder register block
    } // for each output channel
#else
    // !defined(__AVX512__) && !defined(__AVX2__)
    __m128 sum[3][4]; // [3][4]

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
      for (int hbegin = 0; hbegin < 12; hbegin += 3) {
        int hend = hbegin + 3;

#pragma unroll(3)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm_set1_ps(bias[oc]);
          sum[h - hbegin][1] = _mm_set1_ps(bias[oc]);
          sum[h - hbegin][2] = _mm_set1_ps(bias[oc]);
          sum[h - hbegin][3] = _mm_set1_ps(bias[oc]);
        }

        for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
          __m128 w_v = _mm_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(3)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD))), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 4)), sum[h - hbegin][1]);
            sum[h - hbegin][2] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 8)), sum[h - hbegin][2]);
            sum[h - hbegin][3] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 12)), sum[h - hbegin][3]);
          }
        }

#pragma unroll(3)
        for (int h = hbegin; h < hend; ++h) {
          _mm_storeu_ps(output + (oc*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 4, sum[h - hbegin][1]);
          _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 8, sum[h - hbegin][2]);
          ((int *)output)[(oc*WOUT + h)*WOUT + 12] = _mm_extract_ps(sum[h - hbegin][3], 0);
        }
      }

      int hbegin = 12, hend = 13;
#pragma unroll(1)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm_set1_ps(bias[oc]);
        sum[h - hbegin][1] = _mm_set1_ps(bias[oc]);
        sum[h - hbegin][2] = _mm_set1_ps(bias[oc]);
        sum[h - hbegin][3] = _mm_set1_ps(bias[oc]);
      }

      for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
        __m128 w_v = _mm_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(1)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD))), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 4)), sum[h - hbegin][1]);
          sum[h - hbegin][2] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 8)), sum[h - hbegin][2]);
          sum[h - hbegin][3] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 12)), sum[h - hbegin][3]);
        }
      }

#pragma unroll(1)
      for (int h = hbegin; h < hend; ++h) {
        _mm_storeu_ps(output + (oc*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 4, sum[h - hbegin][1]);
        _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 8, sum[h - hbegin][2]);
        ((int *)output)[(oc*WOUT + h)*WOUT + 12] = _mm_extract_ps(sum[h - hbegin][3], 0);
      }
    }

    for (int b = 1; b < ncolblocks; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
        for (int hbegin = 0; hbegin < 12; hbegin += 3) {
          int hend = hbegin + 3;

#pragma unroll(3)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT);
            sum[h - hbegin][1] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT + 4);
            sum[h - hbegin][2] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT + 8);
            sum[h - hbegin][3] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT + 12);
          }

          for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
            __m128 w_v = _mm_set1_ps(values[j]);
            int off = colidx[j];

#pragma unroll(3)
            for (int h = hbegin; h < hend; ++h) {
              sum[h - hbegin][0] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD))), sum[h - hbegin][0]);
              sum[h - hbegin][1] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 4)), sum[h - hbegin][1]);
              sum[h - hbegin][2] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 8)), sum[h - hbegin][2]);
              sum[h - hbegin][3] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 12)), sum[h - hbegin][3]);
            }
          }

#pragma unroll(3)
          for (int h = hbegin; h < hend; ++h) {
            _mm_storeu_ps(output + (oc*WOUT + h)*WOUT, sum[h - hbegin][0]);
            _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 4, sum[h - hbegin][1]);
            _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 8, sum[h - hbegin][2]);
            ((int *)output)[(oc*WOUT + h)*WOUT + 12] = _mm_extract_ps(sum[h - hbegin][3], 0);
          }
        }

        int hbegin = 12, hend = 13;
#pragma unroll(1)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT);
          sum[h - hbegin][1] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT + 4);
          sum[h - hbegin][2] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT + 8);
          sum[h - hbegin][3] = _mm_loadu_ps(output + (oc*WOUT + h)*WOUT + 12);
        }

        for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
          __m128 w_v = _mm_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(1)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD))), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 4)), sum[h - hbegin][1]);
            sum[h - hbegin][2] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 8)), sum[h - hbegin][2]);
            sum[h - hbegin][3] = _mm_add_ps(_mm_mul_ps(w_v, _mm_loadu_ps(input + off + h*(WIDTH + PAD) + 12)), sum[h - hbegin][3]);
          }
        }

#pragma unroll(1)
        for (int h = hbegin; h < hend; ++h) {
          _mm_storeu_ps(output + (oc*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 4, sum[h - hbegin][1]);
          _mm_storeu_ps(output + (oc*WOUT + h)*WOUT + 8, sum[h - hbegin][2]);
          ((int *)output)[(oc*WOUT + h)*WOUT + 12] = _mm_extract_ps(sum[h - hbegin][3], 0);
        }
      } // for each oc
    } // for each col block
#endif
  }

  conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;
}

// JSP: Overfeat each group of conv3
// Input: 256 x 13 x 13 => 676 B per channel, 169 KB total
// Output: 512 x 12 x 12 => 576 B per channel, 288 KB total
// Weight: 512 x 256 x 3 x 3 => 36B per channel pair, 9 KB per output channel, 18 KB per input channel, 4.5 MB total
//         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
// FLOPS: 2 x 512 x 256 x 12 x 12 x 3 x 3 = 324 MFLOPS

// Conv4
// Input: 512 x 13 x 13 => 338 KB total
// Output: 1024 x 12 x 12 => 576 KB total

// Conv5
// Input: 1024 x 13 x 13 => 676 KB total
// Output: 1024 x 12 x 12 => 576 KB total

static /*inline*/ void __attribute__((noinline)) sconv345_overfeat(
    // input features
    const float *input,
    // weights
    const int **rowptr_blocked, const int **colidx_blocked, const float **values_blocked,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int out_channels,
    float *scratch) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WIDTH = 12;
  const int WOUT = 12;
  const int PAD = 1;

  int nthread_groups = nthreads;
#if 0 // def __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups /= 2; // 1 group per core in Xeon
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  int c_per_thread = (out_channels/OC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
  int c_begin = std::min(c_per_thread*tid_in_group, out_channels/OC_BLOCK);
  int c_end = std::min(c_begin + c_per_thread, out_channels/OC_BLOCK);

#ifndef __AVX512F__
  __declspec(aligned(64)) int mask_temp[8] = { -1, -1, -1, -1, 0, 0, 0, 0 };
  __m256i mask_v = _mm256_load_si256((__m256i *)mask_temp);
#endif

  for (int oc_begin = c_begin*OC_BLOCK; oc_begin < c_end*OC_BLOCK; oc_begin += OC_BLOCK) {
#ifdef __AVX512F__
    __m512 sum[12];

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    int hbegin = 0, hend = 12;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
      __m512 bias_v = _mm512_set1_ps(bias[oc]);

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = bias_v;
      }

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

#define W_PREFETCH_DISTANCE (1)

//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
      }
    } // for each oc channel

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
        int jbegin = rowptr[oc];
        int jend = rowptr[oc + 1];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
          //_mm_prefetch((const char *)(scratch + ((oc - oc_begin + 1)*WOUT + h)*16), _MM_HINT_T0);
//          _mm_prefetch((const char *)(values + jend + (h - hbegin)*16), _MM_HINT_T0);
//          _mm_prefetch((const char *)(colidx + jend + (h - hbegin)*16), _MM_HINT_T0);
        }

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(12)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin]);
          }
        }

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
        }
      } // for each out channel
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
        _mm_prefetch((const char *)(output + oc*WOUT*WOUT + h*16), _MM_HINT_T0);
      }

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

//      _mm_prefetch((const char *)(values + rowptr[oc + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(values + rowptr[oc + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[oc + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[oc + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_mask_storeu_ps(output + (oc*WOUT + h)*WOUT, 0x0fff, sum[h - hbegin]);
      }
    }
#else
    __m256 sum[(WOUT + 1)/2][2]; // [6][2]
    __m256 w_v;
    int off;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
      __m256 bias_v = _mm256_set1_ps(bias[oc]);

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(6) // compiler gives warning for unroll pragma, but it still unrolls as we want.
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }
    }

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {

        // Upper half of images
        int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8);
        }

        int jbegin = rowptr[oc];
        int jend = rowptr[oc + 1];

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(6)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }

        // Lower half of images
        hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8);
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(6)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }
      }
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8);
      }

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (oc*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (oc*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((oc - oc_begin)*WOUT + h)*16 + 8);
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (oc*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (oc*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }
    }
#endif
  }

  conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;
}

/**
 * This version has fewer unaligned loads but involves extra instructions of shifting within vector registers
 */
static /*inline*/ void __attribute__((noinline)) sconv345_split(
    // input features
    const float *input,
    // weights
    const int *rowptr,
    const int *colidx,
    const float *values,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int out_channels,
    float *scratch) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WIDTH = 13;
  const int WOUT = 13;
  const int PAD = 1;

  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups /= 2; // 1 group per core in Xeon
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  int c_per_thread = (out_channels/OC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
  int c_begin = std::min(c_per_thread*tid_in_group, out_channels/OC_BLOCK);
  int c_end = std::min(c_begin + c_per_thread, out_channels/OC_BLOCK);

#ifndef __AVX512F__
  __declspec(aligned(64)) int mask_temp[8] = { -1, -1, -1, -1, -1, 0, 0, 0 };
  __m256i mask_v = _mm256_load_si256((__m256i *)mask_temp);
#endif


  for (int oc_begin = c_begin*OC_BLOCK; oc_begin < c_end*OC_BLOCK; oc_begin += OC_BLOCK) {
#if 1 // def __AVX512F__
    int b = 0;
    __m512 sum[13];

    int hbegin = 0, hend = WOUT;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
      __m512 bias_v = _mm512_set1_ps(bias[oc]);
#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = bias_v;
      }

      int jbegin = rowptr[(b*out_channels + oc)*3], jend = rowptr[(b*out_channels + oc)*3 + 1];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 1], jend = rowptr[(b*out_channels + oc)*3 + 2];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 2], jend = rowptr[(b*out_channels + oc)*3 + 3];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
      }
    } // for each oc channel

    for (b = 1; b < ncolblocks - 1; ++b) {
      hbegin = 0, hend = WOUT;
      for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_maskz_expandloadu_ps(0xfffc, scratch + ((oc - oc_begin)*WOUT + h)*16);
        }

        int jbegin = rowptr[(b*out_channels + oc)*3], jend = rowptr[(b*out_channels + oc)*3 + 1];

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(13)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
          }
        }

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
        }

        jbegin = rowptr[(b*out_channels + oc)*3 + 1], jend = rowptr[(b*out_channels + oc)*3 + 2];

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(13)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
          }
        }

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
        }

        jbegin = rowptr[(b*out_channels + oc)*3 + 2], jend = rowptr[(b*out_channels + oc)*3 + 3];

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(13)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
          }
        }

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
        }
      } // for each oc
    } // for each col block

    hbegin = 0; hend = WOUT;
    b = ncolblocks - 1;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_expandloadu_ps(0xfffc, scratch + ((oc - oc_begin)*WOUT + h)*16);
        _mm_prefetch((const char *)(output + oc*WOUT*WOUT + h*16), _MM_HINT_T0);
      }

      int jbegin = rowptr[(b*out_channels + oc)*3], jend = rowptr[(b*out_channels + oc)*3 + 1];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 1], jend = rowptr[(b*out_channels + oc)*3 + 2];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 2], jend = rowptr[(b*out_channels + oc)*3 + 3];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(13)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(13)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_mask_storeu_ps(output + (oc*WOUT + h)*WOUT, 0x1fff, sum[h - hbegin]);
      }
    } // for each oc
#else
    __m256 sum[(WOUT + 1)/2][2]; // [7][2]
    __m256 w_v;
    int off;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {
      __m256 bias_v = _mm256_set1_ps(bias[out_channel]);

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7) // compiler gives warning for unroll pragma, but it still unrolls as we want.
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }
    }

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {

        // Upper half of images
        int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
        }

        int jbegin = rowptr[out_channel];
        int jend = rowptr[out_channel + 1];

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(7)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }

        // Lower half of images
        hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(6)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }
      }
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
      }

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }
    }
#endif
  }

  conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;
}

static /*inline*/ void __attribute__((noinline)) sconv345_split_overfeat(
    // input features
    const float *input,
    // weights
    const int *rowptr,
    const int *colidx,
    const float *values,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int out_channels,
    float *scratch) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WIDTH = 12;
  const int WOUT = 12;
  const int PAD = 1;

  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups /= 2; // 1 group per core in Xeon
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  int c_per_thread = (out_channels/OC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
  int c_begin = std::min(c_per_thread*tid_in_group, out_channels/OC_BLOCK);
  int c_end = std::min(c_begin + c_per_thread, out_channels/OC_BLOCK);

#ifndef __AVX512F__
  __declspec(aligned(64)) int mask_temp[8] = { -1, -1, -1, -1, -1, 0, 0, 0 };
  __m256i mask_v = _mm256_load_si256((__m256i *)mask_temp);
#endif

  for (int oc_begin = c_begin*OC_BLOCK; oc_begin < c_end*OC_BLOCK; oc_begin += OC_BLOCK) {
#if 1 // def __AVX512F__
    int b = 0;
    __m512 sum[WOUT];

    int hbegin = 0, hend = WOUT;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
      __m512 bias_v = _mm512_set1_ps(bias[oc]);
#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = bias_v;
      }

      int jbegin = rowptr[(b*out_channels + oc)*3], jend = rowptr[(b*out_channels + oc)*3 + 1];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 1], jend = rowptr[(b*out_channels + oc)*3 + 2];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 2], jend = rowptr[(b*out_channels + oc)*3 + 3];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
      }
    } // for each oc channel

    for (b = 1; b < ncolblocks - 1; ++b) {
      hbegin = 0, hend = WOUT;
      for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_maskz_expandloadu_ps(0xfffc, scratch + ((oc - oc_begin)*WOUT + h)*16);
        }

        int jbegin = rowptr[(b*out_channels + oc)*3], jend = rowptr[(b*out_channels + oc)*3 + 1];

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(12)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
          }
        }

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
        }

        jbegin = rowptr[(b*out_channels + oc)*3 + 1], jend = rowptr[(b*out_channels + oc)*3 + 2];

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(12)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
          }
        }

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
        }

        jbegin = rowptr[(b*out_channels + oc)*3 + 2], jend = rowptr[(b*out_channels + oc)*3 + 3];

        for (int j = jbegin; j < jend; ++j) {
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];

#pragma unroll(12)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
          }
        }

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          _mm512_store_ps(scratch + ((oc - oc_begin)*WOUT + h)*16, sum[h - hbegin]);
        }
      } // for each oc
    } // for each col block

    hbegin = 0; hend = WOUT;
    b = ncolblocks - 1;

    for (int oc = oc_begin; oc < oc_begin + OC_BLOCK; ++oc) {
#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_expandloadu_ps(0xfffc, scratch + ((oc - oc_begin)*WOUT + h)*16);
        _mm_prefetch((const char *)(output + oc*WOUT*WOUT + h*16), _MM_HINT_T0);
      }

      int jbegin = rowptr[(b*out_channels + oc)*3], jend = rowptr[(b*out_channels + oc)*3 + 1];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 1], jend = rowptr[(b*out_channels + oc)*3 + 2];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin] = _mm512_maskz_compress_ps(0xfffe, sum[h - hbegin]);
      }

      jbegin = rowptr[(b*out_channels + oc)*3 + 2], jend = rowptr[(b*out_channels + oc)*3 + 3];

      for (int j = jbegin; j < jend; ++j) {
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];

#pragma unroll(12)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin] = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + h*16), sum[h - hbegin]);
        }
      }

#pragma unroll(12)
      for (int h = hbegin; h < hend; ++h) {
        _mm512_mask_storeu_ps(output + (oc*WOUT + h)*WOUT, 0x0fff, sum[h - hbegin]);
      }
    } // for each oc
#else
    __m256 sum[(WOUT + 1)/2][2]; // [7][2]
    __m256 w_v;
    int off;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {
      __m256 bias_v = _mm256_set1_ps(bias[out_channel]);

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7) // compiler gives warning for unroll pragma, but it still unrolls as we want.
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }
    }

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {

        // Upper half of images
        int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
        }

        int jbegin = rowptr[out_channel];
        int jend = rowptr[out_channel + 1];

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(7)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }

        // Lower half of images
        hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(6)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }
      }
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int out_channel = oc_begin; out_channel < oc_begin + OC_BLOCK; ++out_channel) {

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
      }

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - oc_begin)*WOUT + h)*16 + 8);
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }
    }
#endif
  }

  conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;
}

/**
 * Default un-optimized sparse convolution implementation
 */
inline void caffe_cpu_sconv_default(
    // input features
    const float *input_padded, int in_channels,
    int height, int width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    // weights
    const int *rowptr, const int *colidx, const float *values,
    int kernel_h, int kernel_w,
    const float *bias,
    // output features
    float *output,
    int out_channels)
{
  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  conv_cycles_of_this_batch[omp_get_thread_num()*16] = __rdtsc();

  if (dilation_h != 1 || dilation_w != 1) {
    for (int output_row = 0; output_row < output_h; ++output_row) {
      for (int output_col = 0; output_col < output_w; ++output_col) {

        for (int oc = 0; oc < out_channels; ++oc) {
          float sum = bias[oc];

          for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
            int off = colidx[j];

            int kernel_col = off%(width + pad_w);
            int kernel_row = (off/(width + pad_w))%(height + pad_h);
            int in_channel = off/((width + pad_w)*(height + pad_h));

            int input_row = kernel_row * dilation_h + output_row * stride_h;
            int input_col = kernel_col * dilation_w + output_col * stride_w;

            sum += values[j]*input_padded[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
          }

          output[(oc * output_h + output_row) * output_w + output_col] = sum;
        }
      }
    }
  }
  else {
    for (int output_row = 0; output_row < output_h; ++output_row) {
      for (int output_col = 0; output_col < output_w; ++output_col) {

        const float *in = input_padded + output_row * stride_h * (width + pad_w) + output_col * stride_w;

        for (int oc = 0; oc < out_channels; ++oc) {
          float sum = bias[oc];
#ifdef DBG_SCONV
          if (oc == 359 && output_row == 32 && output_col == 28) {
            printf("%g", sum);
          }
#endif

          for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
            assert(in + colidx[j] >= input_padded && in + colidx[j] < input_padded + in_channels*(width + pad_w)*(height + pad_h) + pad_h*(width + 2*pad_w));
            sum += values[j]*in[colidx[j]];
#ifdef DBG_SCONV
            if (oc == 359 && output_row == 32 && output_col == 28) {
              printf(" + %g*%d:%g:%g", values[j], colidx[j], in[colidx[j]], sum);
            }
#endif
          }

          output[(oc*output_h + output_row)*output_w + output_col] = sum;
#ifdef DBG_SCONV
          if (oc == 359 && output_row == 32 && output_col == 28) {
            printf(" = %g\n", sum);
          }
#endif
        }
      }
    }
  }

  conv_cycles_of_this_batch[omp_get_thread_num()*16] = __rdtsc() - conv_cycles_of_this_batch[omp_get_thread_num()*16];
}

#endif /* _CAFFE_UTIL_CONV_HPP_ */
