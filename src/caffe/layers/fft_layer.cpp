#include <vector>

#include "caffe/layers/fft_layer.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

template <typename Dtype>
void FFTLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void FFTLayer<Dtype>::WeightAlign() {
  BaseConvolutionLayer<Dtype>::WeightAlign();

  int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];
  int stride_h = this->stride_.cpu_data()[0], stride_w = this->stride_.cpu_data()[1];
  int dilation_h = this->dilation_.cpu_data()[0], dilation_w = this->dilation_.cpu_data()[1];

  if (stride_h != 1 || stride_w != 1 || dilation_h != 1 || dilation_w != 1) {
    LOG(FATAL) << "non-unit stride or dilation";
  }
  if (kernel_h != kernel_w) {
    LOG(FATAL) << "kernel_h != kernel_w";
  }

  int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

  fft_height_ = height + std::max(2*pad_h, kernel_h - 1);
  fft_width_ = width + std::max(2*pad_w, kernel_w - 1);

  // 1. Transform weights

  // transform weights to Winograd domain
//  Dtype* weight_orig = new Dtype[this->blobs_[0]->count()];

  vector<int> shape;
  shape.push_back(this->conv_out_channels_);
  int ch_gr = (this->conv_in_channels_/this->group_);
  shape.push_back(ch_gr);
  shape.push_back(fft_height_);
  shape.push_back(2*(fft_width_/2 + 1)); /* 2x for complex type */
  fft_weight_.Reshape(shape);
  fft_weight_conj_.Reshape(shape);
  Dtype *fft_weight_data = fft_weight_.mutable_cpu_data();
  Dtype *fft_weight_conj_data = fft_weight_conj_.mutable_cpu_data();

  int in_N[2];
  in_N[0] = fft_height_;
  in_N[1] = fft_width_;

  int out_N[2];
  out_N[0] = shape[2];
  out_N[1] = shape[3]/2;

  int in_N_inplace[2];
  in_N_inplace[0] = shape[2];
  in_N_inplace[1] = shape[3];

  fft_map_real_size_ = fft_height_*fft_width_;
  fft_map_complex_size_ = out_N[0]*out_N[1];

  void *fft_weight_handle = caffe_cpu_fft_plan_many_dft_r2c<Dtype>(
      2, /* dimension */
      in_N, /* size */
      fft_weight_.count(0, 2), /* howmany */
      fft_weight_data, /* input */
      in_N_inplace, /* inembed */
      1, /* istride */
      in_N_inplace[0]*in_N_inplace[1], /* idist */
      (std::complex<Dtype> *)fft_weight_data, /* out */
      out_N, /* onembed */
      1, /* ostride */
      fft_map_complex_size_, /* odist */
      FFTW_ESTIMATE);

  caffe_memset(fft_weight_.count()*sizeof(Dtype), 0, fft_weight_.mutable_cpu_data());

  const Dtype* weight_data = this->blobs_[0]->cpu_data();
//#pragma omp parallel for collapse(2)
  for (int n = 0; n < this->conv_out_channels_; ++n) {
    for (int c = 0; c < ch_gr; ++c) {
      for (int r = 0; r < kernel_h; ++r) {
        for (int s = 0; s < kernel_w; ++s) {
          fft_weight_data[fft_weight_.offset(n, c, r, s)] =
              weight_data[((n*ch_gr + c)*kernel_h + r)*kernel_w + s];
        }
      }
    }
  }

  caffe_cpu_fft_execute<Dtype>(fft_weight_handle);

  // conjugate weights because we're actually doing correlation not convolution
  for (int n = 0; n < this->conv_out_channels_; ++n) {
    for (int c = 0; c < ch_gr; ++c) {
      for (int y = 0; y < shape[2]; ++y) {
        for (int x = 0; x < shape[3]/2; ++x) {
          fft_weight_conj_data[(((y*shape[3]/2 + x)*this->conv_out_channels_ + n)*ch_gr + c)*2] =
              fft_weight_data[fft_weight_.offset(n, c, y, 2*x)];
          fft_weight_conj_data[(((y*shape[3]/2 + x)*this->conv_out_channels_ + n)*ch_gr + c)*2 + 1] =
              -fft_weight_data[fft_weight_.offset(n, c, y, 2*x + 1)];
        }
      }
    }
  }

  for (int n = 0; n < this->conv_out_channels_; ++n) {
    for (int c = 0; c < ch_gr; ++c) {
      for (int y = 0; y < shape[2]; ++y) {
        for (int x = 0; x < shape[3]/2; ++x) {
          fft_weight_data[(((y*shape[3]/2 + x)*this->conv_out_channels_ + n)*ch_gr + c)*2] =
              fft_weight_conj_data[(((y*shape[3]/2 + x)*this->conv_out_channels_ + n)*ch_gr + c)*2];
          fft_weight_data[(((y*shape[3]/2 + x)*this->conv_out_channels_ + n)*ch_gr + c)*2 + 1] =
              -fft_weight_conj_data[(((y*shape[3]/2 + x)*this->conv_out_channels_ + n)*ch_gr + c)*2 + 1];
        }
      }
    }
  }

//  for (int n = 0; n < 1; ++n) {
//    for (int c = 0; c < 1; ++c) {
//      for (int r = 0; r < kernel_h; ++r) {
//        for (int s = 0; s < kernel_w; ++s) {
//          fprintf(stderr, "%g ", weight_orig[((n*ch_gr + c)*kernel_h + r)*kernel_w + s]);
//        }
//        fprintf(stderr, "\n");
//      }
//    }
//  }
//
//  for (int n = 0; n < 1; ++n) {
//    for (int c = 0; c < 1; ++c) {
//      for (int y = 0; y < shape[2]; ++y) {
//        for (int x = 0; x < shape[3]/2; ++x) {
//          fprintf(
//              stderr, "(%g, %g) ",
//              fft_weight_.cpu_data()[fft_weight_.offset(n, c, y, 2*x)],
//              fft_weight_.cpu_data()[fft_weight_.offset(n, c, y, 2*x + 1)]);
//        }
//        fprintf(stderr, "\n");
//      }
//    }
//  }

  // 2. Plan input activation transformation
  fft_map_in_real_ = new Dtype[fft_map_real_size_*this->conv_in_channels_];
  fft_map_in_complex_ = new std::complex<Dtype>[fft_map_complex_size_*this->conv_in_channels_];
  fft_map_out_complex_ = new std::complex<Dtype>[fft_map_complex_size_*this->conv_out_channels_];
  fft_map_out_real_ = new Dtype[fft_map_real_size_*this->conv_out_channels_];

  fft_handle_ = caffe_cpu_fft_plan_many_dft_r2c<Dtype>(
      2, /* dimension */
      in_N, /* size */
      this->conv_in_channels_, /* howmany */
      fft_map_in_real_, /* input */
      in_N, /* inembed */
      1, /* istride */
      fft_map_real_size_, /* idist */
      fft_map_in_complex_, /* out */
      out_N, /* onembed */
      1, /* ostride */
      fft_map_complex_size_, /* odist */
      FFTW_ESTIMATE);

  fft_back_handle_ = caffe_cpu_fft_plan_many_dft_r2c<Dtype>(
      2, /* dimension */
      in_N, /* size */
      this->conv_out_channels_, /* howmany */
      fft_map_in_real_, /* input */
      in_N, /* inembed */
      1, /* istride */
      fft_map_real_size_, /* idist */
      fft_map_in_complex_, /* out */
      out_N, /* onembed */
      1, /* ostride */
      fft_map_complex_size_, /* odist */
      FFTW_ESTIMATE);

  // 3. Plan output activation inverse transformation
  ifft_handle_ = caffe_cpu_fft_plan_many_dft_c2r<Dtype>(
      2, /* dimension */
      in_N, /* size */
      this->conv_out_channels_, /* howmany */
      fft_map_out_complex_, /* input */
      out_N, /* inembed */
      1, /* istride */
      fft_map_complex_size_, /* idist */
      fft_map_out_real_, /* out */
      in_N, /* onembed */
      1, /* ostride */
      fft_map_real_size_, /* odist */
      FFTW_ESTIMATE);

  ifft_back_handle_ = caffe_cpu_fft_plan_many_dft_c2r<Dtype>(
      2, /* dimension */
      in_N, /* size */
      this->conv_in_channels_, /* howmany */
      fft_map_out_complex_, /* input */
      out_N, /* inembed */
      1, /* istride */
      fft_map_complex_size_, /* idist */
      fft_map_out_real_, /* out */
      in_N, /* onembed */
      1, /* ostride */
      fft_map_real_size_, /* odist */
      FFTW_ESTIMATE);
}

template <>
void FFTLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template <>
void FFTLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  const float* weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const float* bottom_data = bottom[i]->cpu_data();
    float* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      for (int c = 0; c < this->conv_in_channels_; ++c) {
        int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
        int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

        const float *map_in = bottom_data + n*this->bottom_dim_ + c*height*width;

        // 0-padding: map_in --> fft_map_in_real_
        caffe_memset(fft_map_real_size_*sizeof(float), 0, fft_map_in_real_);
        for (int y = 0; y < height; ++y) {
          memcpy(fft_map_in_real_ + (c*fft_height_ + y + pad_h)*fft_width_ + pad_w, map_in + y*width, width*sizeof(float));
        }
      }

      // FFT: fft_map_in_real_ --> fft_map_in_complex_
      caffe_cpu_fft_execute<float>(fft_handle_);

      std::complex<float> alpha = 1, beta = 0;
      for (int j = 0; j < fft_map_complex_size_; ++j) {
        for (int g = 0; g < this->group_; ++g) {
          cblas_cgemv(
              CblasRowMajor, CblasNoTrans,
              this->conv_out_channels_/this->group_, this->conv_in_channels_/this->group_,
              &alpha,
              ((const std::complex<float> *)fft_weight_conj_.cpu_data()) + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_),
              this->conv_in_channels_/this->group_,
              fft_map_in_complex_ + g*this->conv_in_channels_/this->group_*fft_map_complex_size_ + j, fft_map_complex_size_,
              &beta,
              fft_map_out_complex_ + g*this->conv_out_channels_/this->group_*fft_map_complex_size_ + j, fft_map_complex_size_);
        }
      }

      caffe_cpu_fft_execute<float>(ifft_handle_);

      // IFFT: map_out_complex --> map_out_real
      const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];
      float ifft_scale = 1./((float) fft_map_real_size_);
      for (int c = 0; c < this->conv_out_channels_; c++) {
        float *map_out = top_data + n*this->top_dim_ + c*output_h*output_w;

        // post-process: map_out_real --> map_out
        for (int y = 0; y < std::min(output_h, fft_height_); y++) {
          for (int x = 0; x < std::min(output_w, fft_width_); x++) {
            map_out[y*output_w + x] =
              ifft_scale*fft_map_out_real_[(c*fft_height_ + y)*fft_width_ + x];
          }
        }
      } // for each output channel

//      if (this->layer_param().name() == "conv1" && n == 0) {
//        const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];
//        for (int c = 0; c < this->conv_out_channels_; ++c) {
//          for (int y = 0; y < output_h; ++y) {
//            for (int x = 0; x < output_w; ++x) {
//              fprintf(stderr, "%g ", top_data[n*this->top_dim_ + (c*output_h + y)*output_w + x]);
//            }
//            fprintf(stderr, "\n");
//          }
//          fprintf(stderr, "\n");
//        }
//      }

      if (this->bias_term_) {
        const float* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n*this->top_dim_, bias);
      }
    } // for each image
  }
}

template <>
void FFTLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <>
void FFTLayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  const float* weight = this->blobs_[0]->cpu_data();
  float* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const float* top_diff = top[i]->cpu_diff();
    const float* bottom_data = bottom[i]->cpu_data();
    float* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      float* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
//          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
//              bottom_diff + n * this->bottom_dim_);

          const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];
          for (int c = 0; c < this->conv_out_channels_; ++c) {
            int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

            const float *map_in = top_diff + n*this->top_dim_ + c*output_h*output_w;

            // 0-padding: map_in --> fft_map_in_real_
            caffe_memset(fft_map_real_size_*sizeof(float), 0, fft_map_in_real_);
            for (int y = 0; y < output_h; ++y) {
              memcpy(fft_map_in_real_ + (c*fft_height_ + y)*fft_width_, map_in + y*output_w, output_w*sizeof(float));
            }
          }

          // FFT: fft_map_in_real_ --> fft_map_in_complex_
          caffe_cpu_fft_execute<float>(fft_back_handle_);

          std::complex<float> alpha = 1, beta = 0;
          for (int j = 0; j < fft_map_complex_size_; ++j) {
            for (int g = 0; g < this->group_; ++g) {
              cblas_cgemv(
                  CblasRowMajor, CblasNoTrans,
                  this->conv_in_channels_/this->group_, this->conv_out_channels_/this->group_,
                  &alpha,
                  ((const std::complex<float> *)fft_weight_.cpu_data()) + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_),
                  this->conv_out_channels_/this->group_,
                  fft_map_in_complex_ + g*this->conv_out_channels_/this->group_*fft_map_complex_size_ + j, fft_map_complex_size_,
                  &beta,
                  fft_map_out_complex_ + g*this->conv_in_channels_/this->group_*fft_map_complex_size_ + j, fft_map_complex_size_);
            }
          }

          caffe_cpu_fft_execute<float>(ifft_back_handle_);

          // IFFT: map_out_complex --> map_out_real
          float ifft_scale = 1./((float)fft_map_real_size_);
          for (int c = 0; c < this->conv_in_channels_; c++) {
            int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
            float *map_out = bottom_diff + n*this->bottom_dim_ + c*height*width;

            // post-process: map_out_real --> map_out
            for (int y = 0; y < height; y++) {
              for (int x = 0; x < width; x++) {
                map_out[y*width + x] =
                  ifft_scale*fft_map_out_real_[(c*fft_height_ + y)*fft_width_ + x];
              }
            }
          } // for each output channel
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FFTLayer);
#endif

INSTANTIATE_CLASS(FFTLayer);
REGISTER_LAYER_CLASS(FFT);

}  // namespace caffe
