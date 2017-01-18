#include <vector>

#include "caffe/layers/winograd_layer.hpp"
#include "caffe/util/winograd.hpp"

namespace caffe {

template <typename Dtype>
__global__ void winograd_input_im2col_gpu_kernel(
  const int n,
  const Dtype *data, Dtype *col_buff,
  int height, int width,
  int pad_h, int pad_w,
  int ntiles_h, int ntiles_w,
  int tile_h_in, int tile_w_in,
  int tile_h_out, int tile_w_out)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_in;
    const int y = index/tile_w_in%tile_h_in;
    const int tile_w = index/tile_w_in/tile_h_in%ntiles_w;
    const int tile_h = index/tile_w_in/tile_h_in/ntiles_w%ntiles_h;
    const int c = index/tile_w_in/tile_h_in/ntiles_w/ntiles_h;

    int in_y = tile_h*tile_h_out + y - pad_h;
    int in_x = tile_w*tile_w_out + x - pad_w;

    if (in_y < 0 || in_x < 0 || in_y >= height || in_x >= width) {
      col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x] = 0;
    }
    else {
      col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x] = data[(c*height + in_y)*width + in_x];
    }
  }
}

template <typename Dtype>
__global__ void winograd_output_col2im_gpu_kernel(
  const int n,
  const Dtype *col_buff, Dtype *data,
  int output_h, int output_w,
  int ntiles_h, int ntiles_w,
  int tile_h_out, int tile_w_out)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_out;
    const int y = index/tile_w_out%tile_h_out;
    const int tile_w = index/tile_w_out/tile_h_out%ntiles_w;
    const int tile_h = index/tile_w_out/tile_h_out/ntiles_w%ntiles_h;
    const int c = index/tile_w_out/tile_h_out/ntiles_w/ntiles_h;

    int out_y = tile_h*tile_h_out + y;
    int out_x = tile_w*tile_w_out + x;

    if (out_y < output_h && out_x < output_w) {
      data[(c*output_h + out_y)*output_w + out_x] =
          col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x];
    }
  }
}

template <typename Dtype>
__global__ void winograd_output_im2col_gpu_kernel(
  const int n,
  const Dtype *data, Dtype *col_buff,
  int output_h, int output_w,
  int ntiles_h, int ntiles_w,
  int tile_h_out, int tile_w_out)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_out;
    const int y = index/tile_w_out%tile_h_out;
    const int tile_w = index/tile_w_out/tile_h_out%ntiles_w;
    const int tile_h = index/tile_w_out/tile_h_out/ntiles_w%ntiles_h;
    const int c = index/tile_w_out/tile_h_out/ntiles_w/ntiles_h;

    int out_y = tile_h*tile_h_out + y;
    int out_x = tile_w*tile_w_out + x;

    if (out_y < 0 || out_x < 0 || out_y >= output_h || out_x >= output_w) {
      col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x] = 0;
    }
    else {
      col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x] =
          data[(c*output_h + out_y)*output_w + out_x];
    }
  }
}

template <typename Dtype>
__global__ void winograd_input_col2im_gpu_kernel(
  const int n,
  const Dtype *col_buff, Dtype *data,
  int height, int width,
  int pad_h, int pad_w,
  int ntiles_h, int ntiles_w,
  int tile_h_in, int tile_w_in,
  int tile_h_out, int tile_w_out)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_in;
    const int y = index/tile_w_in%tile_h_in;
    const int tile_w = index/tile_w_in/tile_h_in%ntiles_w;
    const int tile_h = index/tile_w_in/tile_h_in/ntiles_w%ntiles_h;
    const int c = index/tile_w_in/tile_h_in/ntiles_w/ntiles_h;

    int in_y = tile_h*tile_h_out + y - pad_h;
    int in_x = tile_w*tile_w_out + x - pad_w;

    if (in_y >= 0 && in_x >= 0 && in_y < height && in_x < width) {
      data[(c*height + in_y)*width + in_x] +=
          col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x];
    }
  }
}

template <>
void WinogradLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

//#define PROFILE_WINOGRAD

template <>
void WinogradLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<float> *AKronA = WinogradAKronA<float>::getInstance(kernel_h);
  WinogradBKronB<float> *BKronB = WinogradBKronB<float>::getInstance(kernel_h);
  WinogradGKronG<float> *GKronG = WinogradGKronG<float>::getInstance(kernel_h);

  const float* weight = this->blobs_[0]->gpu_data();

#ifdef PROFILE_WINOGRAD
  CPUTimer timer;
#endif

  for (int i = 0; i < bottom.size(); ++i) {
    const float* bottom_data = bottom[i]->gpu_data();
    float* top_data = top[i]->mutable_gpu_data();

    int M = this->conv_in_channels_*ntiles_h_*ntiles_w_;
    int num_kernels = this->num_*this->conv_in_channels_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;
    int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
    int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

#ifdef PROFILE_WINOGRAD
    timer.Start();
#endif
    winograd_input_im2col_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                              CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, bottom_data, temp2_.mutable_gpu_data(),
      height, width,
      pad_h, pad_w,
      ntiles_h_, ntiles_w_,
      tile_h_in_, tile_w_in_,
      tile_h_out_, tile_w_out_);
    CUDA_POST_KERNEL_CHECK;
#ifdef PROFILE_WINOGRAD
    LOG(INFO) << "winograd_input_im2col takes " << timer.MicroSeconds()/1e6;
#endif

    // Transform input to Winograd domain
#ifdef PROFILE_WINOGRAD
    timer.Start();
#endif
    caffe_gpu_gemm<float>(CblasTrans, CblasTrans,
      tile_h_in_*tile_w_in_, this->num_*M, tile_h_in_*tile_w_in_,
      (float)1, BKronB->get()->gpu_data(), temp2_.mutable_gpu_data(),
      (float)0, temp1_.mutable_gpu_data());
      // temp1_ has (tile_h_in*tile_w_in) x num_ x (conv_in_channels) x (ntiles_h*ntiles_w) dimension
#ifdef PROFILE_WINOGRAD
    LOG(INFO) << "Transformation of bottom takes " << timer.MicroSeconds()/1e6;
#endif

#ifdef PROFILE_WINOGRAD
    timer.Start();
#endif
    // Convolution in Winograd domain
    {
      float alpha = 1, beta = 0;

      int M = this->conv_out_channels_/this->group_;
      int N = ntiles_h_*ntiles_w_;
      int K = this->conv_in_channels_/this->group_;

      if (!weight_ptrs_initialized_) {
        float **weight_ptrs = (float **)weight_ptrs_->mutable_cpu_data();
        for (int n = 0; n < this->num_; ++n) {
          for (int j = 0; j < tile_h_in_*tile_w_in_*this->group_; ++j) {
            weight_ptrs[n*tile_h_in_*tile_w_in_*this->group_ + j] = 
              this->blobs_[0]->mutable_gpu_data() +
              j*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_);
          }
        }
        weight_ptrs_initialized_ = true;
      }

      // TODO: instead of tile_h_in_ x tile_w_in_ x num_ instances of 
      // N x C x (ntiles_h_*ntiles_w_) GEMMs,
      // use tile_h_in_ x tile_w_in_ instances of
      // N x C x (num_*ntiles_h_*ntiles_w_) GEMMs
      CUBLAS_CHECK(cublasSgemmBatched(
        Caffe::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const float **)in_activation_ptrs_->gpu_data(), N,
        (const float **)weight_ptrs_->gpu_data(), K,
        &beta,
        (float **)out_activation_ptrs_->mutable_gpu_data(), N,
        in_activation_ptrs_->count()));
    }
      // col_buff has num_ x (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w)
#ifdef PROFILE_WINOGRAD
    LOG(INFO) << "Convolution takes " << timer.MicroSeconds()/1e6;
#endif

    // Transform back to time domain
#ifdef PROFILE_WINOGRAD
    timer.Start();
#endif
    caffe_gpu_gemm<float>(CblasTrans, CblasNoTrans,
        this->num_*this->conv_out_channels_*ntiles_h_*ntiles_w_, tile_h_out_*tile_w_out_, tile_h_in_*tile_w_in_,
        (float)1, temp2_.gpu_data(), AKronA->get()->gpu_data(),
        (float)0, temp1_.mutable_gpu_data());
#ifdef PROFILE_WINOGRAD
    LOG(INFO) << "Inverse transformation of top takes " << timer.MicroSeconds()/1e6;
#endif

#ifdef PROFILE_WINOGRAD
    timer.Start();
#endif
    num_kernels = this->num_*this->conv_out_channels_*ntiles_h_*ntiles_w_*tile_h_out_*tile_w_out_;
    const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];
    winograd_output_col2im_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                               CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels,
      temp1_.gpu_data(), top_data,
      output_h, output_w,
      ntiles_h_, ntiles_w_,
      tile_h_out_, tile_w_out_); 
    CUDA_POST_KERNEL_CHECK;
#ifdef PROFILE_WINOGRAD
    LOG(INFO) << "winograd_output_col2im takes " << timer.MicroSeconds()/1e6;
#endif

    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      if (this->bias_term_) {
        const float* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <>
void WinogradLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <>
void WinogradLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<float> *AKronA = WinogradAKronA<float>::getInstance(kernel_h);
  WinogradBKronB<float> *BKronB = WinogradBKronB<float>::getInstance(kernel_h);
  WinogradGKronG<float> *GKronG = WinogradGKronG<float>::getInstance(kernel_h);

  const float* weight = this->blobs_[0]->gpu_data();
  float* weight_diff = this->blobs_[0]->mutable_gpu_diff();

	/*const float *weight_cpu = this->blobs_[0]->cpu_data();
  fprintf(stderr, "weight_winograd\n");
  for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
    for (int n = 0; n < this->conv_out_channels_; ++n) {
      for (int c = 0; c < this->conv_in_channels_; ++c) {
        fprintf(stderr, "%g ", weight_cpu[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
      }
    }
    fprintf(stderr, "\n");
  }*/

#ifdef PROFILE_WINOGRAD
  CPUTimer timer;
#endif

  for (int i = 0; i < top.size(); ++i) {
    const float* top_diff = top[i]->gpu_diff();
    const float* bottom_data = bottom[i]->gpu_data();
    float* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      float* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      int M = this->conv_out_channels_*ntiles_h_*ntiles_w_;
      int num_kernels = this->num_*this->conv_out_channels_*ntiles_h_*ntiles_w_*tile_h_out_*tile_w_out_;
      const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];
      const int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
      const int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      winograd_output_im2col_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                                 CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels,
        top_diff, temp1_.mutable_gpu_data(),
        output_h, output_w,
        ntiles_h_, ntiles_w_,
        tile_h_out_, tile_w_out_);
      CUDA_POST_KERNEL_CHECK;
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "winograd_output_im2col takes " << timer.MicroSeconds()/1e6;
#endif

      // Transform out_diff to Winograd domain
#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans,
          tile_h_in_*tile_w_in_, this->num_*M, tile_h_out_*tile_w_out_,
          (float)1, AKronA->get()->gpu_data(), temp1_.mutable_gpu_data(),
          (float)0, temp2_.mutable_gpu_data());
      // temp2_ has (tile_h_in*tile_w_in) x num_ x (conv_out_channels) x (ntiles_h*ntiles_w) dimension
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "Transformation of top_diff takes " << timer.MicroSeconds()/1e6;
#endif

      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        int num_kernels = this->num_*this->conv_in_channels_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;

        winograd_input_im2col_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                                  CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels, bottom_data, this->col_buffer_.mutable_gpu_data(),
          height, width,
          pad_h, pad_w,
          ntiles_h_, ntiles_w_,
          tile_h_in_, tile_w_in_,
          tile_h_out_, tile_w_out_);
        CUDA_POST_KERNEL_CHECK;
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "winograd_input_im2col takes " << timer.MicroSeconds()/1e6;
#endif

        // Transform input to Winograd domain
#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        caffe_gpu_gemm<float>(CblasTrans, CblasTrans,
            tile_h_in_*tile_w_in_, this->num_*this->conv_in_channels_*ntiles_h_*ntiles_w_, tile_h_in_*tile_w_in_,
            (float)1, BKronB->get()->gpu_data(), this->col_buffer_.mutable_gpu_data(),
            (float)0, temp1_.mutable_gpu_data());
        // temp1_ has (tile_h_in*tile_w_in) x num_ x (conv_in_channels) x (ntiles_h*ntiles_w) dimension
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "Transformation of bottom takes " << timer.MicroSeconds()/1e6;
#endif

        if (false/*n == 0*/) {
          const float *weight_diff_cpu = this->blobs_[0]->cpu_diff();
          fprintf(stderr, "weight_diff_winograd0[0]\n");
          for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
            for (int n = 0; n < this->conv_out_channels_; ++n) {
              for (int c = 0; c < this->conv_in_channels_; ++c) {
                fprintf(stderr, "%g ", weight_diff_cpu[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
              }
            }
            fprintf(stderr, "\n");
          }
        }

#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif

        if (!weight_diff_ptrs_initialized_) {
          float **weight_diff_ptrs = (float **)weight_diff_ptrs_->mutable_cpu_data();
          for (int j = 0; j < tile_h_in_*tile_w_in_*this->group_; ++j) {
            weight_diff_ptrs[j] =
              this->blobs_[0]->mutable_gpu_diff() +
              j*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_);
          }
          weight_diff_ptrs_initialized_ = true;
        }
        
        for (int n = 0; n < this->num_; ++n) {
          float alpha = 1, beta = 1;

          int M = this->conv_out_channels_/this->group_;
          int N = this->conv_in_channels_/this->group_;
          int K = ntiles_h_*ntiles_w_;

          CUBLAS_CHECK(cublasSgemmBatched(
            Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            (const float **)in_activation_ptrs_->gpu_data() + n*tile_h_in_*tile_w_in_*this->group_, K,
            (const float **)out_activation_ptrs_->gpu_data() + n*tile_h_in_*tile_w_in_*this->group_, K,
            &beta,
            (float **)weight_diff_ptrs_->mutable_gpu_data(), N,
            tile_h_in_*tile_w_in_*this->group_));
            // weight_diff has (tile_h_in*tile_w_in) x (conv_out_channels) x (conv_in_channels/group) dimension
          
#if 0
          const float *weight_diff_cpu = this->blobs_[0]->cpu_diff();
          for (int i = 0; i < tile_h_in_*tile_w_in_*this->conv_out_channels_*(this->conv_in_channels_/this->group_); ++i) {
            if (isnan(weight_diff_cpu[i])) {
              ostringstream str;
              str << "nan at weight_diff[" << i << "]";
              LOG(FATAL) << str.str();
            }
          }
#endif          

          if (false/*n == this->num_ - 1*/) {
            float *temp_weight = NULL;
            size_t len = this->conv_out_channels_*(this->conv_in_channels_/this->group_)*kernel_h*kernel_w;
            CUDA_CHECK(cudaMalloc(&temp_weight, sizeof(float)*len));

            caffe_gpu_gemm<float>(CblasTrans, CblasNoTrans,
                this->conv_out_channels_*(this->conv_in_channels_/this->group_), kernel_h*kernel_w, tile_h_in_*tile_w_in_,
                (float)1, weight_diff, GKronG->get()->gpu_data(),
                (float)0, temp_weight);
                
            float *temp_weight_cpu = new float[len];
            CUDA_CHECK(cudaFree(temp_weight));
            CUDA_CHECK(cudaMemcpy(temp_weight_cpu, temp_weight, sizeof(float)*len, cudaMemcpyDeviceToHost));

            fprintf(stderr, "weight_diff[%d]\n", n);
            for (int m = 0; m < this->conv_out_channels_; ++m) {
              for (int c = 0; c < this->conv_in_channels_/this->group_; ++c) {
                for (int i = 0; i < kernel_h*kernel_w; ++i) {
                  fprintf(stderr, "%g ", temp_weight_cpu[(m*(this->conv_in_channels_/this->group_) + c)*kernel_h*kernel_w + i]);
                }
              }
              fprintf(stderr, "\n");
            }
            delete[] temp_weight_cpu;

            const float *weight_diff_cpu = this->blobs_[0]->cpu_diff();
            fprintf(stderr, "weight_diff_winograd[%d]\n", n);
            for (int n = 0; n < this->conv_out_channels_; ++n) {
              for (int c = 0; c < this->conv_in_channels_; ++c) {
                for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
                  fprintf(stderr, "%g ", weight_diff_cpu[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
                }
              }
              fprintf(stderr, "\n");
            }
          }
        } // for each input
        
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "Convolution for weight gradient takes " << timer.MicroSeconds()/1e6;
#endif
      } // param_propagate_down_[0]

      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        // Convolution in Winograd domain
        float alpha = 1, beta = 0;
        int M = this->conv_in_channels_/this->group_;
        int N = ntiles_h_*ntiles_w_;
        int K = this->conv_out_channels_/this->group_;

        CUBLAS_CHECK(cublasSgemmBatched(
          Caffe::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
          N, M, K,
          &alpha,
          (const float **)out_activation_ptrs_->gpu_data(), N,
          (const float **)weight_ptrs_->gpu_data(), M,
          &beta,
          (float **)in_activation_ptrs_->mutable_gpu_data(), N,
          in_activation_ptrs_->count()));
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "Convolution for bottom gradient takes " << timer.MicroSeconds()/1e6;
#endif

        // Transform back to time domain
#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        caffe_gpu_gemm<float>(CblasTrans, CblasTrans,
            this->num_*this->conv_in_channels_*ntiles_h_*ntiles_w_, tile_h_in_*tile_w_in_, tile_h_in_*tile_w_in_,
            (float)1, temp1_.mutable_gpu_data(), BKronB->get()->gpu_data(),
            (float)0, this->col_buffer_.mutable_gpu_data());
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "Inverse transformation of bottom_diff takes " << timer.MicroSeconds()/1e6;
#endif

#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        num_kernels = this->num_*this->conv_in_channels_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;

        CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(float)*this->num_*this->conv_in_channels_*height*width));
        winograd_input_col2im_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                                  CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels,
          this->col_buffer_.gpu_data(), bottom_diff,
          height, width,
          pad_h, pad_w,
          ntiles_h_, ntiles_w_,
          tile_h_in_, tile_w_in_,
          tile_h_out_, tile_w_out_);
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "winograd_input_col2im takes " << timer.MicroSeconds()/1e6;
#endif

#if 0
        const float *bottom_diff_cpu = bottom[i]->cpu_diff();
        for (int i = 0; i < this->bottom_dim_; ++i) {
          if (isnan(bottom_diff_cpu[i])) {
            ostringstream str;
            str << "nan at bottom_diff[" << n << ", " << i << "]";
          }
        }
#endif
      } // propagate_down_[i]
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WinogradLayer);

}  // namespace caffe
