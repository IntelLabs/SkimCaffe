#ifndef _CAFFE_UTIL_POOL_HPP_
#define _CAFFE_UTIL_POOL_HPP_

template<typename Dtype, int STRIDE_H, int STRIDE_W, int KERNEL_H, int KERNEL_W, int PAD_H, int PAD_W, int HEIGHT, int WIDTH>
void pool_(const Dtype *input, Dtype *output, int *mask_output, Dtype min = -FLT_MAX)
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
      Dtype maximum = min;
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

      Dtype maximum = min;
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

    for ( ; pw < (WIDTH + PAD_W - KERNEL_W + STRIDE_W - 1)/STRIDE_W; ++pw) {
      int wstart = pw*STRIDE_W - PAD_W;
      int wend = wstart + KERNEL_W;
      assert(wstart >= 0);
      assert(wend <= WIDTH);

      Dtype maximum = min;
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

    for ( ; pw < pooled_width; ++pw) {
      int wstart = pw*STRIDE_W - PAD_W;
      int wend = WIDTH;
      assert(wstart >= 0);
      assert(wstart + KERNEL_W >= WIDTH);

      Dtype maximum = min;
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
      Dtype maximum = min;
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
