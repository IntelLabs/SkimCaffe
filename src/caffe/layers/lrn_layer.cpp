#include <vector>
#include <omp.h>
#include <immintrin.h>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/intrinsic.hpp"

namespace caffe {

template <typename Dtype>
LRNLayer<Dtype>::~LRNLayer()
{
  free(scale_temp_);
  free(padded_square_);
}

template <typename Dtype>
void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    split_top_vec_.clear();
    split_top_vec_.push_back(&product_input_);
    split_top_vec_.push_back(&square_input_);
    LayerParameter split_param;
    split_layer_.reset(new SplitLayer<Dtype>(split_param));
    split_layer_->SetUp(bottom, split_top_vec_);
    // Set up square_layer_ to square the inputs.
    square_bottom_vec_.clear();
    square_top_vec_.clear();
    square_bottom_vec_.push_back(&square_input_);
    square_top_vec_.push_back(&square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(Dtype(2));
    square_layer_.reset(new PowerLayer<Dtype>(square_param));
    square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    pool_top_vec_.clear();
    pool_top_vec_.push_back(&pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(pre_pad_);
    pool_param.mutable_pooling_param()->set_kernel_size(size_);
    pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
    pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
    // the sum of a squared neighborhood (the output of pool_layer_).
    power_top_vec_.clear();
    power_top_vec_.push_back(&power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-beta_);
    power_param.mutable_power_param()->set_scale(alpha_);
    power_param.mutable_power_param()->set_shift(Dtype(1));
    power_layer_.reset(new PowerLayer<Dtype>(power_param));
    power_layer_->SetUp(pool_top_vec_, power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    product_bottom_vec_.clear();
    product_bottom_vec_.push_back(&product_input_);
    product_bottom_vec_.push_back(&power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
    product_layer_->SetUp(product_bottom_vec_, top);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(num_, channels_, height_, width_);
    scale_.Reshape(num_, channels_, height_, width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    split_layer_->Reshape(bottom, split_top_vec_);
    square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
    pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
    power_layer_->Reshape(pool_top_vec_, power_top_vec_);
    product_layer_->Reshape(product_bottom_vec_, top);
    break;
  }
  if (channels_*height_*width_%8 != 0) {
    LOG(FATAL) << "channels_*height_*width_ should be a multiple of 8";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <>
void LRNLayer<double>::CrossChannelForward_cpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template<typename Dtype, int CHANNELS, int SIZE, int HEIGHT, int WIDTH>
void lrn_(Dtype *top_data, Dtype alpha, Dtype beta, Dtype k, Dtype *padded_square_data, const Dtype *bottom_data, int bottom_offset, int offset)
{
  assert(WIDTH%VLEN == 0);

  float alpha_over_size = alpha/SIZE;
  const int PRE_PAD = (SIZE - 1)/2;

  SIMDFPTYPE scale_data_v[(WIDTH + VLEN - 1)/VLEN];
  SIMDFPTYPE alpha_over_size_v = _MM_SET1(alpha/SIZE);

  for (int i = 0; i < HEIGHT; ++i) {
    // compute the padded square
    for (int c = PRE_PAD; c < SIZE; ++c) {
      for (int j = 0; j < WIDTH; j += VLEN) {
        SIMDFPTYPE d = _MM_LOAD(bottom_data + bottom_offset + ((c - PRE_PAD)*HEIGHT + i)*WIDTH + j);
        _MM_STORE(padded_square_data + c*WIDTH + j, _MM_MUL(d, d));
      }
    }

    // Create the first channel scale
    for (int j = 0; j < WIDTH; j += VLEN) {
      scale_data_v[j/VLEN] = _MM_FMADD(
          alpha_over_size_v,
          _MM_LOAD(padded_square_data + j),
          _MM_SET1(k));
    }
    for (int c = 1; c < SIZE; ++c) {
      for (int j = 0; j < WIDTH; j += VLEN) {
        scale_data_v[j/VLEN] = _MM_FMADD(
            _MM_SET1(alpha_over_size),
            _MM_LOAD(padded_square_data + c*WIDTH + j),
            scale_data_v[j/VLEN]);

        SIMDFPTYPE v = _MM_POW(scale_data_v[j/VLEN], _MM_SET1(-beta));
        v = _MM_MUL(v, _MM_LOAD(bottom_data + offset + i*WIDTH + j));
        _MM_STORE(top_data + offset + i*WIDTH + j, v);
      }
    }

    for (int c = 1; c < CHANNELS; ++c) {
      if (c < CHANNELS + PRE_PAD - SIZE + 1) {
#pragma unroll(WIDTH/VLEN)
        for (int j = 0; j < WIDTH; j += VLEN) {
          SIMDFPTYPE d = _MM_LOAD(bottom_data + bottom_offset + ((c - PRE_PAD + SIZE - 1)*HEIGHT + i)*WIDTH + j);
          _MM_STORE(padded_square_data + (c + SIZE - 1)%(SIZE + 1)*WIDTH + j, _MM_MUL(d, d));
        }
      }

#pragma unroll(WIDTH/VLEN)
      for (int j = 0; j < WIDTH; j += VLEN) {
        scale_data_v[j/VLEN] = _MM_FMADD(
            alpha_over_size_v,
            _MM_SUB(
                _MM_LOAD(padded_square_data + (c + SIZE - 1)%(SIZE + 1)*WIDTH + j),
                _MM_LOAD(padded_square_data + (c - 1)%(SIZE + 1)*WIDTH + j)),
            scale_data_v[j/VLEN]);

        SIMDFPTYPE v = _MM_POW(scale_data_v[j/VLEN], _MM_SET1(-beta));
        v = _MM_MUL(v, _MM_LOAD(bottom_data + offset + (c*HEIGHT + i)*WIDTH + j));
        _MM_STORE(top_data + offset + (c*HEIGHT + i)*WIDTH + j, v);
      }
    }
  }
}

template <>
void LRNLayer<float>::CrossChannelForward_cpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  if (!padded_square_) {
    posix_memalign(
        (void **)&padded_square_,
        4096,
        sizeof(float) * omp_get_max_threads() * (channels_ + size_ - 1) * width_);
  }
  float alpha_over_size = alpha_ / size_;
  if (!scale_temp_) {
    posix_memalign(
        (void **)&scale_temp_,
        4096,
        sizeof(float) * omp_get_max_threads() * channels_ * height_ * width_);
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    float* padded_square_data = padded_square_ + tid * (channels_ + size_ - 1) * width_;
    for (int i = 0; i < pre_pad_ * width_; ++i) {
      padded_square_data[i] = 0;
    }
    for (int i = (channels_ + pre_pad_) * width_; i < (channels_ + size_ - 1) * width_; ++i) {
      padded_square_data[i] = 0;
    }

    float *scale_data = scale_temp_ + tid * channels_ * height_ * width_;

    // go through the images
#pragma omp for
    for (int n = 0; n < num_; ++n) {
      int bottom_offset = bottom[0]->offset(n);
      int offset = scale_.offset(n, 0);

#ifndef __AVX512F__
      if (5 == size_ && height_ == width_ && 1 == k_) {
        if (56 == height_) {
          // GoogLeNet
          if (64 == channels_) {
            lrn_<float, 64, 5, 56, 56>(top_data, alpha_, beta_, 1.f, padded_square_data, bottom_data, bottom_offset, offset);
            continue;
          }
          else if (192 == channels_) {
            lrn_<float, 192, 5, 56, 56>(top_data, alpha_, beta_, 1.f, padded_square_data, bottom_data, bottom_offset, offset);
            continue;
          }
        }
      }
#endif

      for (int i = 0; i < height_; ++i) {
        // compute the padded square
        for (int c = pre_pad_; c < channels_ + pre_pad_; ++c) {
          for (int j = 0; j < width_; ++j) {
            float d = bottom_data[bottom_offset + (c - pre_pad_) * height_ * width_ + i * width_ + j];
            padded_square_data[c * width_ + j] = d * d;
          }
        }

        // Create the first channel scale
        for (int j = 0; j < width_; ++j) {
          scale_data[i * width_ + j] = k_ + alpha_over_size*padded_square_data[j];
        }
        for (int c = 1; c < size_; ++c) {
          for (int j = 0; j < width_; ++j) {
            scale_data[i * width_ + j] += alpha_over_size*padded_square_data[c * width_ + j];
          }
        }

        for (int c = 1; c < channels_; ++c) {
          for (int j = 0; j < width_; ++j) {
            scale_data[(c * height_ + i) * width_ + j] =
              scale_data[((c - 1) * height_ + i) * width_ + j] +
              alpha_over_size*(
                  padded_square_data[(c + size_ - 1) * width_ + j] -
                  padded_square_data[(c - 1) * width_ + j]);
          }
        }
      }

#ifdef __AVX2__
      for (int i = 0; i < channels_ * height_ * width_; i += 8) {
        __m256 v = _mm256_pow_ps(_mm256_load_ps(scale_data + i), _mm256_set1_ps(-beta_));
        v = _mm256_mul_ps(v, _mm256_load_ps(bottom_data + offset + i));
        _mm256_store_ps(top_data + offset + i, v);
      }
#else
      for (int i = 0; i < channels_ * height_ * width_; i += 8) {
        __m128 v = _mm_pow_ps(_mm_load_ps(scale_data + i), _mm_set1_ps(-beta_));
        v = _mm_mul_ps(v, _mm_load_ps(bottom_data + offset + i));
        _mm_store_ps(top_data + offset + i, v);
        v = _mm_pow_ps(_mm_load_ps(scale_data + i + 4), _mm_set1_ps(-beta_));
        v = _mm_mul_ps(v, _mm_load_ps(bottom_data + offset + i + 4));
        _mm_store_ps(top_data + offset + i + 4, v);
      }
#endif
    }
  } // omp parallel
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, square_top_vec_);
  pool_layer_->Forward(square_top_vec_, pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = scale_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
  Blob<Dtype> accum_ratio(1, 1, height_, width_);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
  // We hack a little bit by using the diff() to store an additional result
  Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
  caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
  Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
  caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

  // go through individual data
  int inverse_pre_pad = size_ - (size_ + 1) / 2;
  for (int n = 0; n < num_; ++n) {
    int block_offset = scale_.offset(n);
    // first, compute diff_i * y_i / s_i
    caffe_mul<Dtype>(channels_ * height_ * width_,
        top_diff + block_offset, top_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    caffe_div<Dtype>(channels_ * height_ * width_,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
        scale_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    // Now, compute the accumulated ratios and the bottom diff
    caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
    for (int c = 0; c < size_ - 1; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
    for (int c = 0; c < channels_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
          accum_ratio_data);
      // compute bottom diff
      caffe_mul<Dtype>(height_ * width_,
          bottom_data + top[0]->offset(n, c),
          accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy<Dtype>(height_ * width_, -cache_ratio_value,
          accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c));
      caffe_axpy<Dtype>(height_ * width_, -1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    vector<bool> product_propagate_down(2, true);
    product_layer_->Backward(top, product_propagate_down, product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->Backward(square_top_vec_, propagate_down,
                            square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LRNLayer);
STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(LRNLayer);

}  // namespace caffe
