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

//#define VECTORIZE_OVER_INPUTS

//static const int COL_MAJOR_IC_BLOCK = 8;
//static const int COL_MAJOR_OC_BLOCK = 64;

extern synk::Barrier *barriers[256];

extern unsigned long long conv_cycles_of_this_batch[1024*16], transpose_cycle, pool_cycle;

static int get_col_major_ic_block(int nnz, int num_out_channels, int num_in_channels) {
  // # of in-channels to have on average 32 non-zeros per out-channel
  double nnz_per_oc_and_ic = (double)nnz/num_out_channels/num_in_channels;
  int ret = std::max(8, 1 << (int)round(log2(std::max(1., 32/nnz_per_oc_and_ic))));
  ret = std::min(num_in_channels/2, ret);
    // if block size is bigger than num_in_channels/2, we will have only 1 block
    // but our sconv kernels need at least 2 blocks.
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
template<int WIDTH, int K>
static /*inline*/ void __attribute__((noinline)) sconv_unit_stride(
    // input features
    const float *input,
    // weights
    const int **rowptr_blocked, const int **colidx_blocked, const float **values_blocked,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int num_out_channels,
    float *scratch) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  assert(ncolblocks >= 2);

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WOUT = WIDTH;
  const int PAD = (K - 1)/2;

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

  int num_oc_blocks = (num_out_channels + OC_BLOCK - 1)/OC_BLOCK;
  int oc_blocks_per_thread = (num_oc_blocks + nthreads_per_group - 1)/nthreads_per_group;
  int oc_block_begin = std::min(oc_blocks_per_thread*tid_in_group, num_oc_blocks);
  int oc_block_end = std::min(oc_block_begin + oc_blocks_per_thread, num_oc_blocks);
  const int ALIGNED_W = (WIDTH + 16 - 1)/16*16;

#ifdef __AVX512F__
  const int REG_BLOCK_SIZE = 30; // use at most 30 SIMD registers out of 32
#else
  const int REG_BLOCK_SIZE = 14; // use at most 14 SIMD registers out of 16
#endif

  const int REG_BLOCK_W = (WIDTH + VLEN - 1)/VLEN;
  assert(REG_BLOCK_W <= REG_BLOCK_SIZE);
  const int REG_BLOCK_H = WIDTH < REG_BLOCK_SIZE/REG_BLOCK_W ? WIDTH : REG_BLOCK_SIZE/REG_BLOCK_W;
  // WIDTH = 13 (AlexNet conv3-5), AVX2 : REG_BLOCK_W = 2, REG_BLOCK_H = 7, ALIGNED_W = 16
  // WIDTH = 56 (GoogLeNet), AVX2 : REG_BLOCK_W = 7, REG_BLOCK_H = 2, ALIGNED_W = 64

#ifdef __AVX512F__
  __mmask16 mask_v = (1 << (WIDTH%VLEN)) - 1;
#else
  __declspec(aligned(64)) int mask_temp[VLEN] = { 0 };
  for (int i = 0; i < WIDTH%VLEN; ++i) {
    mask_temp[i] = -1;
  }
  SIMDITYPE mask_v = _MM_LOAD_SI((SIMDITYPE *)mask_temp);
#endif

  for (int oc_begin = oc_block_begin*OC_BLOCK; oc_begin < oc_block_end*OC_BLOCK; oc_begin += OC_BLOCK) {
    int oc_end = std::min(oc_begin + OC_BLOCK, num_out_channels);

    SIMDFPTYPE sum[REG_BLOCK_H][REG_BLOCK_W];
    SIMDFPTYPE w_v;
    int off;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int oc = oc_begin; oc < oc_end; ++oc) {
      SIMDFPTYPE bias_v = _MM_SET1(bias[oc]);

      int jbegin = rowptr[oc];
      int jend = rowptr[oc + 1];

      // register blocking over input image positions
      int hbegin;
      for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
        int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H) // compiler gives warning for unroll pragma, but it still unrolls as we want.
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            sum[h - hbegin][w] = bias_v;

//#define DBG_SCONV
#ifdef DBG_SCONV
#define CHANNEL_TO_DEBUG (359)
#define ROW_TO_DEBUG (32)
#define COL_TO_DEBUG (28)
            if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG && COL_TO_DEBUG >= w*VLEN && COL_TO_DEBUG < (w + 1)*VLEN) {
              float temp[VLEN];
              _MM_STORE(temp, bias_v);
              printf("%g", temp[COL_TO_DEBUG - w*VLEN]);
            }
#endif
          }
        }

#define SCONV_INNER_PROD \
        for (int j = jbegin; j < jend; ++j) { \
          w_v = _MM_SET1(values[j]); \
          off = colidx[j]; \
 \
_Pragma("unroll(REG_BLOCK_H)") \
          for (int h = 0; h < REG_BLOCK_H; ++h) { /* by some reason, iterating from hbegin to hend prevents icc from unrolling */ \
_Pragma("unroll(REG_BLOCK_W") \
            for (int w = 0; w < REG_BLOCK_W; ++w) { \
              sum[h][w] = _MM_FMADD(w_v, _MM_LOADU(input + off + (h + hbegin)*(WIDTH + PAD) + VLEN*w), sum[h][w]); \
            } \
 \
/*#ifdef DBG_SCONV \
            if (out_channel == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) { \
              float temp[VLEN]; \
              _MM_STORE(temp, sum[h - hbegin][COL_TO_DEBUG/VLEN]); \
              printf(" + %g*%d:%g:%g", values[j], off, input[off + ROW_TO_DEBUG*(WIDTH + PAD) + COL_TO_DEBUG], temp[COL_TO_DEBUG%VLEN]); \
            } \
#endif*/ \
          } \
        }

        SCONV_INNER_PROD;

#pragma unroll(REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            _MM_STORE(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
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

#define SCONV_INNER_PROD_REMAINDER \
        for (int j = jbegin; j < jend; ++j) { \
          w_v = _MM_SET1(values[j]); \
          off = colidx[j]; \
 \
_Pragma("unroll(WOUT%REG_BLOCK_H)") \
          for (int h = hbegin; h < hend; ++h) { \
_Pragma("unroll(REG_BLOCK_W)") \
            for (int w = 0; w < REG_BLOCK_W; ++w) { \
              sum[h - hbegin][w] = _MM_FMADD(w_v, _MM_LOADU(input + off + h*(WIDTH + PAD) + VLEN*w), sum[h - hbegin][w]); \
            } \
          } \
        }

        SCONV_INNER_PROD_REMAINDER;

#pragma unroll(WOUT%REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
          for (int w = 0; w < REG_BLOCK_W; ++w) {
            _MM_STORE(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
          }
        }
      } // remainder register block
    } // for each output channel

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int out_channel = oc_begin; out_channel < oc_end; ++out_channel) {
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
              sum[h - hbegin][w] = _MM_LOAD(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
            }
          }

          SCONV_INNER_PROD;

#pragma unroll(REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _MM_STORE(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
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
              sum[h - hbegin][w] = _MM_LOAD(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
            }
          }

          SCONV_INNER_PROD_REMAINDER;

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _MM_STORE(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
            }
          }
        } // remainder register block
      } // for each output channel
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int out_channel = oc_begin; out_channel < oc_end; ++out_channel) {
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
            sum[h - hbegin][w] = _MM_LOAD(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
          }
        }

        SCONV_INNER_PROD;

#pragma unroll(REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
          if (WIDTH%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _MM_STOREU(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }

#ifdef DBG_SCONV
            if (out_channel == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) {
              printf(" = %g\n", output[(out_channel*WOUT + h)*WOUT + COL_TO_DEBUG]);
            }
#endif
          }
          else {
            int w;
#pragma unroll(REG_BLOCK_W - 1)
            for (w = 0; w < REG_BLOCK_W - 1; ++w) {
              _MM_STOREU(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }
            _MM_MASK_STORE(output + (out_channel*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
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
            sum[h - hbegin][w] = _MM_LOAD(scratch + ((out_channel - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
          }
        }

        SCONV_INNER_PROD_REMAINDER;

#pragma unroll(WOUT%REG_BLOCK_H)
        for (int h = hbegin; h < hend; ++h) {
          if (WIDTH%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _MM_STOREU(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }
          }
          else {
            int w;
#pragma unroll(REG_BLOCK_W - 1)
            for (w = 0; w < REG_BLOCK_W - 1; ++w) {
              _MM_STOREU(output + (out_channel*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
            }
            _MM_MASK_STORE(output + (out_channel*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
          }
        }
      } // remainder register block
    } // for each output channel
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
          if (oc == CHANNEL_TO_DEBUG && output_row == ROW_TO_DEBUG && output_col == COL_TO_DEBUG) {
            printf("%g", sum);
          }
#endif

          for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
            assert(in + colidx[j] >= input_padded && in + colidx[j] < input_padded + in_channels*(width + pad_w)*(height + pad_h) + pad_h*(width + 2*pad_w));
            sum += values[j]*in[colidx[j]];
#ifdef DBG_SCONV
            if (oc == CHANNEL_TO_DEBUG && output_row == ROW_TO_DEBUG && output_col == COL_TO_DEBUG) {
              printf(" + %g*%d:%g:%g", values[j], colidx[j], in[colidx[j]], sum);
            }
#endif
          }

          output[(oc*output_h + output_row)*output_w + output_col] = sum;
#ifdef DBG_SCONV
          if (oc == CHANNEL_TO_DEBUG && output_row == ROW_TO_DEBUG && output_col == COL_TO_DEBUG) {
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
