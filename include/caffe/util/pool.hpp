#ifndef _CAFFE_UTIL_POOL_HPP_
#define _CAFFE_UTIL_POOL_HPP_

#include "caffe/util/intrinsic.hpp"

template<int STRIDE_H, int STRIDE_W, int KERNEL_H, int KERNEL_W, int PAD_H, int PAD_W, int HEIGHT, int WIDTH>
void pool_(const float *input, float *output, int *mask_output, float min = -FLT_MAX)
{
  int pooled_height =
      static_cast<int>(ceil(static_cast<float>(HEIGHT + 2*PAD_H - KERNEL_H)/STRIDE_H)) + 1;
  int pooled_width =
      static_cast<int>(ceil(static_cast<float>(WIDTH + 2*PAD_W - KERNEL_W)/STRIDE_W)) + 1;

  int ph;
  for (ph = 0; ph < PAD_H/STRIDE_H; ++ph) {
    int hstart = ph*STRIDE_H - PAD_H;
    int hend = hstart + KERNEL_H;
    assert(hstart <= 0);
    assert(hend <= HEIGHT);
    hstart = 0;

    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw*STRIDE_W - PAD_W;
      int wend = std::min(wstart + KERNEL_W, WIDTH);
      wstart = std::max(wstart, 0);
      float maximum = min;
      int mask = -1;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const int index = h*WIDTH + w;
          if (input[index] > maximum) {
            maximum = input[index];
            mask = index;
          }
        }
      }
      const int pool_index = ph*pooled_width + pw;
      output[pool_index] = maximum;
      mask_output[pool_index] = mask;
    }
  }

  for ( ; ph < (HEIGHT + PAD_H - KERNEL_H + STRIDE_H - 1)/STRIDE_H; ++ph) {
    int hstart = ph*STRIDE_H - PAD_H;
    int hend = hstart + KERNEL_H;
    assert(hstart >= 0);
    assert(hend <= HEIGHT);

    int pw;
    for (pw = 0; pw < PAD_W/STRIDE_W; ++pw) {
      int wstart = pw*STRIDE_W - PAD_W;
      int wend = wstart + KERNEL_W;
      assert(wstart <= 0);
      assert(wend <= WIDTH);
      wstart = 0;

      float maximum = min;
      int mask = -1;
#pragma unroll_and_jam(KERNEL_H)
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const int index = h*WIDTH + w;
          if (input[index] > maximum) {
            maximum = input[index];
            mask = index;
          }
        }
      }
      const int pool_index = ph*pooled_width + pw;
      output[pool_index] = maximum;
      mask_output[pool_index] = mask;
    }

#if defined(__AVX2__) && !defined(__AVX512F__)
    if (1 == STRIDE_H) {
      __declspec(aligned(64)) int identity[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
      SIMDITYPE identity_v = _MM_LOAD_SI((SIMDITYPE *)identity);

      for ( ; pw < ((WIDTH + PAD_W - KERNEL_W + STRIDE_W - 1)/STRIDE_W)/VLEN*VLEN; pw += VLEN) {
        int wstart = pw*STRIDE_W - PAD_W;
        int wend = wstart + KERNEL_W;
        assert(wstart >= 0);
        assert(wend <= WIDTH);

        SIMDFPTYPE maximum_v = _MM_SET1(min);
        SIMDFPTYPE mask_v = _MM_SETZERO();
#pragma unroll(KERNEL_H)
        for (int h = hstart; h < hend; ++h) {
#pragma unroll(KERNEL_W)
          for (int w = wstart; w < wend; ++w) {
            const int index = h*WIDTH + w;
            SIMDFPTYPE in_v = _MM_LOAD(input + index);
            SIMDFPTYPE cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
            maximum_v = _mm256_blendv_ps(in_v, maximum_v, cmp_v); // max = in <= max ? max : in
            mask_v = _mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_add_epi32(_MM_SET1_EPI32(index), identity_v)),
                mask_v,
                cmp_v);
          }
        }

        const int pool_index = ph*pooled_width + pw;
        _MM_STORE(output + pool_index, maximum_v);
        _MM_STORE((float *)(mask_output + pool_index), mask_v);
      }
    }
    else if (2 == STRIDE_H) {
      __declspec(aligned(64)) int identity[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
      __m256i identity_v = _mm256_load_si256((__m256i *)identity);

      for ( ; pw < ((WIDTH + PAD_W - KERNEL_W + STRIDE_W - 1)/STRIDE_W)/VLEN*VLEN; pw += VLEN) {
        int wstart = pw*STRIDE_W - PAD_W;
        int wend = wstart + KERNEL_W;
        assert(wstart >= 0);
        assert(wend <= WIDTH);

        SIMDFPTYPE maximum_v = _MM_SET1(min);
        SIMDFPTYPE mask_v = _MM_SETZERO();
#pragma unroll(KERNEL_H)
        for (int h = hstart; h < hend; ++h) {
#pragma unroll(KERNEL_W)
          for (int w = wstart; w < wend; ++w) {
            const int index = h*WIDTH + w;

            SIMDFPTYPE in_v = _mm256_shuffle_ps(
                _MM_LOAD(input + index),
                _MM_LOAD(input + index + VLEN),
                0x88);
            in_v = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(in_v), 0xd8));

            SIMDFPTYPE cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
            maximum_v = _mm256_blendv_ps(in_v, maximum_v, cmp_v); // max = in <= max ? max : in
            mask_v = _mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_add_epi32(_MM_SET1_EPI32(index), identity_v)),
                mask_v,
                cmp_v);
          }
        }

        const int pool_index = ph*pooled_width + pw;
        _MM_STORE(output + pool_index, maximum_v);
        _MM_STORE((float *)(mask_output + pool_index), mask_v);
      }
    }
    else
#endif
    {
      for ( ; pw < (WIDTH + PAD_W - KERNEL_W + STRIDE_W - 1)/STRIDE_W; ++pw) {
        int wstart = pw*STRIDE_W - PAD_W;
        int wend = wstart + KERNEL_W;
        assert(wstart >= 0);
        assert(wend <= WIDTH);

        float maximum = min;
        int mask = -1;
#pragma unroll(KERNEL_H)
        for (int h = hstart; h < hend; ++h) {
#pragma unroll(KERNEL_W)
          for (int w = wstart; w < wend; ++w) {
            const int index = h*WIDTH + w;
            if (input[index] > maximum) {
              maximum = input[index];
              mask = index;
            }
          }
        }
        const int pool_index = ph*pooled_width + pw;
        output[pool_index] = maximum;
        mask_output[pool_index] = mask;
      }
    }

    for ( ; pw < pooled_width; ++pw) {
      int wstart = pw*STRIDE_W - PAD_W;
      int wend = std::min(wstart + KERNEL_W, WIDTH);
      assert(wstart >= 0);

      float maximum = min;
      int mask = -1;
#pragma unroll_and_jam(KERNEL_H)
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const int index = h*WIDTH + w;
          if (input[index] > maximum) {
            maximum = input[index];
            mask = index;
          }
        }
      }
      const int pool_index = ph*pooled_width + pw;
      output[pool_index] = maximum;
      mask_output[pool_index] = mask;
    }
  }

  for ( ; ph < pooled_height; ++ph) {
    int hstart = ph*STRIDE_H - PAD_H;
    int hend = HEIGHT;
    assert(hstart >= 0);
    assert(hstart + KERNEL_H >= HEIGHT);

    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw*STRIDE_W - PAD_W;
      int wend = std::min(wstart + KERNEL_W, WIDTH);
      wstart = std::max(wstart, 0);
      float maximum = min;
      int mask = -1;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const int index = h*WIDTH + w;
          if (input[index] > maximum) {
            maximum = input[index];
            mask = index;
          }
        }
      }
      const int pool_index = ph*pooled_width + pw;
      output[pool_index] = maximum;
      mask_output[pool_index] = mask;
    }
  }
}

#endif // _CAFFE_UTIL_POOL_HPP_
