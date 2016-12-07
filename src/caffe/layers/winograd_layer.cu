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

template <typename Dtype>
void WinogradLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<Dtype> *AKronA = WinogradAKronA<Dtype>::getInstance(kernel_h);
  WinogradBKronB<Dtype> *BKronB = WinogradBKronB<Dtype>::getInstance(kernel_h);
  WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);

  const Dtype* weight = this->blobs_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      int M = this->conv_in_channels_*ntiles_h_*ntiles_w_;

      Dtype *col_buff = this->col_buffer_.mutable_gpu_data();

      int num_kernels = this->conv_in_channels_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;
      int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
      int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

      winograd_input_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                                CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data + n*this->bottom_dim_, col_buff,
        height, width,
        pad_h, pad_w,
        ntiles_h_, ntiles_w_,
        tile_h_in_, tile_w_in_,
        tile_h_out_, tile_w_out_);
      CUDA_POST_KERNEL_CHECK;

      // Transform input to Winograd domain
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans,
          tile_h_in_*tile_w_in_, M, tile_h_in_*tile_w_in_,
          (Dtype)1, BKronB->get()->gpu_data(), col_buff,
          (Dtype)0, temp1_.mutable_gpu_data());
      // temp_ has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w) dimension

      // Convolution in Winograd domain
      for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
        for (int g = 0; g < this->group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
              this->conv_out_channels_/this->group_, ntiles_h_*ntiles_w_, this->conv_in_channels_/this->group_,
              (Dtype)1,
              weight + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_),
              temp1_.gpu_data() + (j*this->group_ + g)*(this->conv_in_channels_/this->group_)*ntiles_h_*ntiles_w_,
              (Dtype)0, col_buff + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*ntiles_h_*ntiles_w_);
        }
      }
      // col_buff has (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w)

      // Transform back to time domain
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->conv_out_channels_*ntiles_h_*ntiles_w_, tile_h_out_*tile_w_out_, tile_h_in_*tile_w_in_,
          (Dtype)1, col_buff, AKronA->get()->gpu_data(),
          (Dtype)0, temp1_.mutable_gpu_data());

      num_kernels = this->conv_out_channels_*ntiles_h_*ntiles_w_*tile_h_out_*tile_w_out_;
      const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];

      winograd_output_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                                 CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels,
        temp1_.gpu_data(), top_data + n*this->top_dim_,
        output_h, output_w,
        ntiles_h_, ntiles_w_,
        tile_h_out_, tile_w_out_); 
      CUDA_POST_KERNEL_CHECK;

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
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
      for (int n = 0; n < this->num_; ++n) {
        int M = this->conv_out_channels_*ntiles_h_*ntiles_w_;

        float *col_buff = this->col_buffer_.mutable_gpu_data();

        int num_kernels = this->conv_out_channels_*ntiles_h_*ntiles_w_*tile_h_out_*tile_w_out_;
        const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];
        const int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
        const int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

        winograd_output_im2col_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                                   CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels,
          top_diff + n*this->top_dim_, col_buff,
          output_h, output_w,
          ntiles_h_, ntiles_w_,
          tile_h_out_, tile_w_out_);
        CUDA_POST_KERNEL_CHECK;

        // Transform out_diff to Winograd domain
        caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans,
            tile_h_in_*tile_w_in_, M, tile_h_out_*tile_w_out_,
            (float)1, AKronA->get()->gpu_data(), col_buff,
            (float)0, temp1_.mutable_gpu_data());
        // temp_ has (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w) dimension

        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {

          int num_kernels = this->conv_in_channels_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;

          winograd_input_im2col_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                                    CAFFE_CUDA_NUM_THREADS>>>(
            num_kernels, bottom_data + n*this->bottom_dim_, col_buff,
            height, width,
            pad_h, pad_w,
            ntiles_h_, ntiles_w_,
            tile_h_in_, tile_w_in_,
            tile_h_out_, tile_w_out_);
          CUDA_POST_KERNEL_CHECK;

          // Transform input to Winograd domain
          caffe_gpu_gemm<float>(CblasTrans, CblasTrans,
              tile_h_in_*tile_w_in_, this->conv_in_channels_*ntiles_h_*ntiles_w_, tile_h_in_*tile_w_in_,
              (float)1, BKronB->get()->gpu_data(), col_buff,
              (float)0, temp2_.mutable_gpu_data());
          // temp_ has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w) dimension

          for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
            for (int g = 0; g < this->group_; ++g) {
              caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans,
                  this->conv_out_channels_/this->group_, this->conv_in_channels_/this->group_, ntiles_h_*ntiles_w_,
                  (float)1,
                  temp1_.gpu_data() + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*ntiles_h_*ntiles_w_,
                  temp2_.gpu_data() + (j*this->group_ + g)*(this->conv_in_channels_/this->group_)*ntiles_h_*ntiles_w_,
                  (float)1, weight_diff + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_));
            }
          }
          // winograd_weight_ has (tile_h_in*tile_w_in) x (conv_out_channels) x (conv_in_channels/group) dimension
        }

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          // Convolution in Winograd domain
          for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
            for (int g = 0; g < this->group_; ++g) {
              caffe_gpu_gemm<float>(CblasTrans, CblasNoTrans,
                  this->conv_in_channels_/this->group_, ntiles_h_*ntiles_w_, this->conv_out_channels_/this->group_,
                  (float)1,
                  weight + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_),
                  temp1_.gpu_data() + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*ntiles_h_*ntiles_w_,
                  (float)0, col_buff + (j*this->group_ + g)*(this->conv_in_channels_/this->group_)*ntiles_h_*ntiles_w_);
            }
          }
          // col_buff has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w)

          // Transform back to time domain
          caffe_gpu_gemm<float>(CblasTrans, CblasTrans,
              this->conv_in_channels_*ntiles_h_*ntiles_w_, tile_h_in_*tile_w_in_, tile_h_in_*tile_w_in_,
              (float)1, col_buff, BKronB->get()->gpu_data(),
              (float)0, temp1_.mutable_gpu_data());

          num_kernels = this->conv_in_channels_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;

          CUDA_CHECK(cudaMemset(bottom_diff + n*this->bottom_dim_, 0, sizeof(float)*this->conv_in_channels_*height*width));
          winograd_input_col2im_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                                    CAFFE_CUDA_NUM_THREADS>>>(
            num_kernels,
            temp1_.gpu_data(), bottom_diff + n*this->bottom_dim_,
            height, width,
            pad_h, pad_w,
            ntiles_h_, ntiles_w_,
            tile_h_in_, tile_w_in_,
            tile_h_out_, tile_w_out_);
        }
      } // for each image
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WinogradLayer);

}  // namespace caffe
