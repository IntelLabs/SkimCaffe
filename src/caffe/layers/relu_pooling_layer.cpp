#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/relu_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/pool.hpp"

namespace caffe {

using std::min;
using std::max;

template <>
void ReLUPoolingLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <>
void ReLUPoolingLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {
  if (this->layer_param_.relu_param().negative_slope() != 0) {
    LOG(FATAL) << "ConvolutionReLUPoolLayer only supports negative_slope == 0";
  }

  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  float* top_mask = NULL;
  int num = bottom[0]->num();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    int* mask = NULL;  // suppress warnings about uninitalized variables
    if (!use_top_mask) mask = this->max_idx_.mutable_cpu_data();
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();

      // The main loop
#pragma omp parallel for collapse(2)
      for (int n = 0; n < num; ++n) {
        for (int c = 0; c < this->channels_; ++c) {
          // compute offset
          const float *bottom_data_cur = bottom_data + bottom[0]->offset(0, 1)*(this->channels_*n + c);
          float *top_data_cur = top_data + top[0]->offset(0, 1)*(this->channels_*n + c);
          int *mask_cur = mask + top[0]->offset(0, 1)*(this->channels_*n + c);
          float *top_mask_cur = top_mask + top[0]->offset(0, 1)*(this->channels_*n + c);

          for (int ph = 0; ph < this->pooled_height_; ++ph) {
            for (int pw = 0; pw < this->pooled_width_; ++pw) {
              int hstart = ph * this->stride_h_ - this->pad_h_;
              int wstart = pw * this->stride_w_ - this->pad_w_;
              int hend = min(hstart + this->kernel_h_, this->height_);
              int wend = min(wstart + this->kernel_w_, this->width_);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              float maximum = 0;
              int mask = -1;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * this->width_ + w;
                  if (bottom_data_cur[index] > maximum) {
                    maximum = bottom_data_cur[index];
                    mask = static_cast<float>(index);
                  }
                }
              }
              const int pool_index = ph * this->pooled_width_ + pw;
              top_data_cur[pool_index] = maximum;
              top_data_cur[pool_index] = mask;
            }
          }
        } // for each channel
      } // for each input layer
    }
    else { // !use_top_mask
      // JSP: typical path, stride=2 kernel=3

      // The main loop
#pragma omp parallel for
      for (int n = 0; n < num; ++n) {
        for (int c = 0; c < this->channels_; ++c) {
          // compute offset
          const float *bottom_data_cur = bottom_data + bottom[0]->offset(0, 1)*(this->channels_*n + c);
          float *top_data_cur = top_data + top[0]->offset(0, 1)*(this->channels_*n + c);
          int *mask_cur = mask + top[0]->offset(0, 1)*(this->channels_*n + c);

          if (this->stride_h_ == this->stride_w_ && this->kernel_h_ == this->kernel_w_ && this->pad_h_ == this->pad_w_ && this->height_ == this->width_) {
            if (3 == this->kernel_w_) {
              if (2 == this->stride_h_ && 3 == this->kernel_w_ && 0 == this->pad_h_) {
                if (112 == this->height_) {
                  pool_<2, 2, 3, 3, 0, 0, 112, 112>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
                else if (56 == this->height_) {
                  pool_<2, 2, 3, 3, 0, 0, 56, 56>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
                else if (28 == this->height_) {
                  pool_<2, 2, 3, 3, 0, 0, 28, 28>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
                else if (14 == this->height_) {
                  pool_<2, 2, 3, 3, 0, 0, 14, 14>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
                // AlexNet
                else if (55 == height_) {
                  pool_<2, 2, 3, 3, 0, 0, 55, 55>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
              }
              else if (1 == this->stride_h_ && 1 == this->pad_h_) {
                if (28 == this->height_) {
                  pool_<1, 1, 3, 3, 1, 1, 28, 28>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
                else if (14 == this->height_) {
                  pool_<1, 1, 3, 3, 1, 1, 14, 14>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
                else if (7 == this->height_) {
                  pool_<1, 1, 3, 3, 1, 1, 7, 7>(bottom_data_cur, top_data_cur, mask_cur, 0.f);
                  continue;
                }
              }
            }
          }

          for (int ph = 0; ph < this->pooled_height_; ++ph) {
            int hstart = ph * this->stride_h_ - this->pad_h_;
            int hend = min(hstart + this->kernel_h_, this->height_);
            hstart = max(hstart, 0);

            for (int pw = 0; pw < this->pooled_width_; ++pw) {
              int wstart = pw * this->stride_w_ - this->pad_w_;
              int wend = min(wstart + this->kernel_w_, this->width_);
              wstart = max(wstart, 0);
              float maximum = 0;
              int mask = -1;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * this->width_ + w;
                  if (bottom_data_cur[index] > maximum) {
                    maximum = bottom_data_cur[index];
                    mask = index;
                  }
                }
              }
              const int pool_index = ph * this->pooled_width_ + pw;
              top_data_cur[pool_index] = maximum;
              mask_cur[pool_index] = mask;
            }
          }
        } // for each channel
      } // for each input layer
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void ReLUPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ReLUPoolingLayer);
#else
template <typename Dtype>
void ReLUPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ReLUPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(ReLUPoolingLayer);
REGISTER_LAYER_CLASS(ReLUPooling);

}  // namespace caffe
