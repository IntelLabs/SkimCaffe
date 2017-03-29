#include <vector>

#include "caffe/layers/winograd_layer.hpp"
#include "caffe/util/winograd.hpp"

namespace caffe {

template <typename Dtype>
void WinogradLayer<Dtype>::compute_output_shape() {
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
bool WinogradLayer<Dtype>::IsReshapedToWinograd() {
  return !(this->blobs_[0]->shape(2) == this->blobs_[0]->shape(3) && (this->blobs_[0]->shape(2) == 3 || this->blobs_[0]->shape(2) == 5));
}

template <typename Dtype>
void WinogradLayer<Dtype>::ReshapeToWinograd() {
  if (!IsReshapedToWinograd()) {
    // not yet reshaped
    vector<int> shape;
    shape.push_back(tile_h_in_);
    shape.push_back(tile_w_in_);
    shape.push_back(this->conv_out_channels_);
    shape.push_back(this->conv_in_channels_/this->group_);
    this->blobs_[0]->Reshape(shape);
  }
}

template <typename Dtype>
void WinogradLayer<Dtype>::WeightAlign() {
  BaseConvolutionLayer<Dtype>::WeightAlign();

  WeightAlignLocal();
}

template <typename Dtype>
void WinogradLayer<Dtype>::WeightAlignLocal() {
  if (!IsReshapedToWinograd()) {
    // transform weights to Winograd domain
    Dtype* weight_orig = new Dtype[this->blobs_[0]->count()];
    memcpy(weight_orig, this->blobs_[0]->cpu_data(), sizeof(Dtype)*this->blobs_[0]->count());

    ReshapeToWinograd();

    int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];
    WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        tile_h_in_*tile_w_in_, (this->conv_in_channels_/this->group_)*this->conv_out_channels_, kernel_h*kernel_w,
        (Dtype)1, GKronG->get()->cpu_data(), weight_orig,
        (Dtype)0, this->blobs_[0]->mutable_cpu_data());
    delete[] weight_orig;
  }
}

template <typename Dtype>
void WinogradLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

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

  WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);

  tile_h_in_ = GKronG->M;
  tile_w_in_ = GKronG->M;
  tile_h_out_ = tile_h_in_ - GKronG->N + 1, tile_w_out_ = tile_w_in_ - GKronG->N + 1;

  int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];
  int output_h = (height + 2*pad_h - dilation_h*(kernel_h - 1) - 1)/stride_h + 1, output_w = (width + 2*pad_w - dilation_w*(kernel_w - 1) - 1)/stride_w + 1;

  // to cover input image: (ntiles_h_ - 1)*tile_h_out_ + (tile_h_in_ - 1) - pad_h >= height - 1 => ntiles_h_ > (height + pad_h - tile_h_in_)/tile_h_out_
  // to cover output image: ntiles_h_ >= output_h/tile_h_out_
  ntiles_h_ = (std::max(height + pad_h - tile_h_in_ + 1, output_h) + tile_h_out_ - 1)/tile_h_out_;
  ntiles_w_ = (std::max(width + pad_w - tile_w_in_ + 1, output_w) + tile_w_out_ - 1)/tile_w_out_;

  // create temporary buffers
  vector<int> shape;
  shape.push_back(this->num_);
  shape.push_back(tile_h_in_*tile_w_in_);
  shape.push_back(std::max(this->conv_in_channels_, this->conv_out_channels_));
  shape.push_back(ntiles_h_*ntiles_w_);

  if (temp1_.shape() != shape) {
    temp1_.Reshape(shape);
    temp2_.Reshape(shape);

    // create arrays to pointers to prepare for cuda batch sgemm
    shape.clear();
    shape.push_back(tile_h_in_);
    shape.push_back(tile_w_in_);
    shape.push_back(this->group_);

    in_activation_ptrs_.reset(new Blob<long>(shape));
    out_activation_ptrs_.reset(new Blob<long>(shape));
    weight_ptrs_.reset(new Blob<long>(shape));
    weight_diff_ptrs_.reset(new Blob<long>(shape));

    Dtype **in_ptrs = (Dtype **)in_activation_ptrs_->mutable_cpu_data();
    Dtype **out_ptrs = (Dtype **)out_activation_ptrs_->mutable_cpu_data();

    for (int j = 0; j < tile_h_in_*tile_w_in_*this->group_; ++j) {
      in_ptrs[j] =
        temp1_.mutable_gpu_data() +
        j*(this->conv_in_channels_/this->group_)*this->num_*ntiles_h_*ntiles_w_;

      out_ptrs[j] =
        temp2_.mutable_gpu_data() +
        j*(this->conv_out_channels_/this->group_)*this->num_*ntiles_h_*ntiles_w_;
    }

    weight_ptrs_initialized_ = false;
    weight_diff_ptrs_initialized_ = false;
  }

  WeightAlignLocal();
}

template<typename Dtype>
void WinogradLayer<Dtype>::winograd_input_im2col_cpu(const Dtype *data, Dtype *col_buff)
{
  int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
  int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

  for (int c = 0; c < this->conv_in_channels_; ++c) {
    for (int tile_h = 0; tile_h < ntiles_h_; ++tile_h) {
      for (int tile_w = 0; tile_w < ntiles_w_; ++tile_w) {
        for (int y = 0; y < tile_h_in_; ++y) {
          for (int x = 0; x < tile_w_in_; ++x) {
            int in_y = tile_h*tile_h_out_ + y - pad_h;
            int in_x = tile_w*tile_w_out_ + x - pad_w;

            if (in_y < 0 || in_x < 0 || in_y >= height || in_x >= width) {
              col_buff[(((c*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_in_ + y)*tile_w_in_ + x] = 0;
            }
            else {
              col_buff[(((c*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_in_ + y)*tile_w_in_ + x] =
                  data[(c*height + in_y)*width + in_x];
            }
          }
        }
      } // for each tile
    } // for each tile
  } // for each input channel
}

template<typename Dtype>
void WinogradLayer<Dtype>::winograd_output_col2im_cpu(const Dtype *col_buff, Dtype *data)
{
  const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];

  for (int n = 0; n < this->conv_out_channels_; ++n) {
    for (int tile_h = 0; tile_h < ntiles_h_; ++tile_h) {
      for (int tile_w = 0; tile_w < ntiles_w_; ++tile_w) {
        for (int y = 0; y < tile_h_out_; ++y) {
          for (int x = 0; x < tile_w_out_; ++x) {
            int out_y = tile_h*tile_h_out_ + y;
            int out_x = tile_w*tile_w_out_ + x;

            if (out_y < output_h && out_x < output_w) {
              data[(n*output_h + out_y)*output_w + out_x] =
                  col_buff[(((n*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_out_ + y)*tile_w_out_ + x];
            }
          }
        }
      } // for each tile
    } // for each tile
  } // for each input channel
}

template<typename Dtype>
void WinogradLayer<Dtype>::winograd_output_im2col_cpu(const Dtype *data, Dtype *col_buff)
{
  const int output_h = this->output_shape_[0], output_w = this->output_shape_[1];

  for (int n = 0; n < this->conv_out_channels_; ++n) {
    for (int tile_h = 0; tile_h < ntiles_h_; ++tile_h) {
      for (int tile_w = 0; tile_w < ntiles_w_; ++tile_w) {
        for (int y = 0; y < tile_h_out_; ++y) {
          for (int x = 0; x < tile_w_out_; ++x) {
            int out_y = tile_h*tile_h_out_ + y;
            int out_x = tile_w*tile_w_out_ + x;

            if (out_y < 0 || out_x < 0 || out_y >= output_h || out_x >= output_w) {
              col_buff[(((n*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_out_ + y)*tile_w_out_ + x] = 0;
            }
            else {
              col_buff[(((n*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_out_ + y)*tile_w_out_ + x] =
                  data[(n*output_h + out_y)*output_w + out_x];
            }
          }
        }
      } // for each tile
    } // for each tile
  } // for each input channel
}

template<typename Dtype>
void WinogradLayer<Dtype>::winograd_input_col2im_cpu(const Dtype *col_buff, Dtype *data)
{
  int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
  int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

  memset(data, 0, sizeof(Dtype)*this->conv_in_channels_*height*width);

  for (int c = 0; c < this->conv_in_channels_; ++c) {
    for (int tile_h = 0; tile_h < ntiles_h_; ++tile_h) {
      for (int tile_w = 0; tile_w < ntiles_w_; ++tile_w) {
        for (int y = 0; y < tile_h_in_; ++y) {
          for (int x = 0; x < tile_w_in_; ++x) {
            int in_y = tile_h*tile_h_out_ + y - pad_h;
            int in_x = tile_w*tile_w_out_ + x - pad_w;

            if (in_y >= 0 && in_x >= 0 && in_y < height && in_x < width) {
              data[(c*height + in_y)*width + in_x] +=
                  col_buff[(((c*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_in_ + y)*tile_w_in_ + x];
            }
          }
        }
      } // for each tile
    } // for each tile
  } // for each input channel
}

//#define PROFILE_WINOGRAD

template <typename Dtype>
void WinogradLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<Dtype> *AKronA = WinogradAKronA<Dtype>::getInstance(kernel_h);
  WinogradBKronB<Dtype> *BKronB = WinogradBKronB<Dtype>::getInstance(kernel_h);
  WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);

  const Dtype* weight = this->blobs_[0]->cpu_data();

#ifdef PROFILE_WINOGRAD
  CPUTimer timer;
#endif

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      int M = this->conv_in_channels_*ntiles_h_*ntiles_w_;

      Dtype *col_buff = this->col_buffer_.mutable_cpu_data();

#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      winograd_input_im2col_cpu(bottom_data + n*this->bottom_dim_, col_buff);
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "winograd_output_im2col takes " << timer.MilliSeconds()/1000;
#endif

      // Transform input to Winograd domain
#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans,
          tile_h_in_*tile_w_in_, M, tile_h_in_*tile_w_in_,
          (Dtype)1, BKronB->get()->cpu_data(), col_buff,
          (Dtype)0, temp1_.mutable_cpu_data());
      // temp_ has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w) dimension
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "Transformation of bottom takes " << timer.MilliSeconds()/1000;
#endif

#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      // Convolution in Winograd domain
      for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
        for (int g = 0; g < this->group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
              this->conv_out_channels_/this->group_, ntiles_h_*ntiles_w_, this->conv_in_channels_/this->group_,
              (Dtype)1,
              weight + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_),
              temp1_.cpu_data() + (j*this->group_ + g)*(this->conv_in_channels_/this->group_)*ntiles_h_*ntiles_w_,
              (Dtype)0, col_buff + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*ntiles_h_*ntiles_w_);
        }
      }
      // col_buff has (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w)
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "Convolution takes " << timer.MilliSeconds()/1000;
#endif

      // Transform back to time domain
#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->conv_out_channels_*ntiles_h_*ntiles_w_, tile_h_out_*tile_w_out_, tile_h_in_*tile_w_in_,
          (Dtype)1, col_buff, AKronA->get()->cpu_data(),
          (Dtype)0, temp1_.mutable_cpu_data());
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "Inverse transformation of top takes " << timer.MilliSeconds()/1000;
#endif

#ifdef PROFILE_WINOGRAD
      timer.Start();
#endif
      winograd_output_col2im_cpu(temp1_.cpu_data(), top_data + n*this->top_dim_);
#ifdef PROFILE_WINOGRAD
      LOG(INFO) << "winograd_output_col2im takes " << timer.MilliSeconds()/1000;
#endif

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <>
void WinogradLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <>
void WinogradLayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<float> *AKronA = WinogradAKronA<float>::getInstance(kernel_h);
  WinogradBKronB<float> *BKronB = WinogradBKronB<float>::getInstance(kernel_h);
  WinogradGKronG<float> *GKronG = WinogradGKronG<float>::getInstance(kernel_h);

  const float* weight = this->blobs_[0]->cpu_data();
  float* weight_diff = this->blobs_[0]->mutable_cpu_diff();

//  fprintf(stderr, "weight_winograd\n");
//  for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
//    for (int n = 0; n < this->conv_out_channels_; ++n) {
//      for (int c = 0; c < this->conv_in_channels_; ++c) {
//        fprintf(stderr, "%g ", weight[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
//      }
//    }
//    fprintf(stderr, "\n");
//  }

#ifdef PROFILE_WINOGRAD
  CPUTimer timer;
#endif

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
        int M = this->conv_out_channels_*ntiles_h_*ntiles_w_;

        float *col_buff = this->col_buffer_.mutable_cpu_data();

#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        winograd_output_im2col_cpu(top_diff + n*this->top_dim_, col_buff);
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "winograd_output_im2col takes " << timer.MilliSeconds()/1000;
#endif

        // Transform out_diff to Winograd domain
#ifdef PROFILE_WINOGRAD
        timer.Start();
#endif
        caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans,
            tile_h_in_*tile_w_in_, M, tile_h_out_*tile_w_out_,
            (float)1, AKronA->get()->cpu_data(), col_buff,
            (float)0, temp1_.mutable_cpu_data());
        // temp_ has (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w) dimension
#ifdef PROFILE_WINOGRAD
        LOG(INFO) << "Transformation of top_diff takes " << timer.MilliSeconds()/1000;
#endif

        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
#ifdef PROFILE_WINOGRAD
          timer.Start();
#endif
          winograd_input_im2col_cpu(bottom_data + n*this->bottom_dim_, col_buff);
#ifdef PROFILE_WINOGRAD
          LOG(INFO) << "winograd_input_im2col takes " << timer.MilliSeconds()/1000;
#endif

          // Transform input to Winograd domain
#ifdef PROFILE_WINOGRAD
          timer.Start();
#endif
          caffe_cpu_gemm<float>(CblasTrans, CblasTrans,
              tile_h_in_*tile_w_in_, this->conv_in_channels_*ntiles_h_*ntiles_w_, tile_h_in_*tile_w_in_,
              (float)1, BKronB->get()->cpu_data(), col_buff,
              (float)0, temp2_.mutable_cpu_data());
          // temp_ has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w) dimension
#ifdef PROFILE_WINOGRAD
          LOG(INFO) << "Transformation of bottom takes " << timer.MilliSeconds()/1000;
#endif

          if (false/*n == 0*/) {
            fprintf(stderr, "weight_diff_winograd0[0]\n");
            for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
              for (int n = 0; n < this->conv_out_channels_; ++n) {
                for (int c = 0; c < this->conv_in_channels_; ++c) {
                  fprintf(stderr, "%g ", weight_diff[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
                }
              }
              fprintf(stderr, "\n");
            }
          }

#ifdef PROFILE_WINOGRAD
          timer.Start();
#endif
          for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
            for (int g = 0; g < this->group_; ++g) {
              caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans,
                  this->conv_out_channels_/this->group_, this->conv_in_channels_/this->group_, ntiles_h_*ntiles_w_,
                  (float)1,
                  temp1_.cpu_data() + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*ntiles_h_*ntiles_w_,
                  temp2_.cpu_data() + (j*this->group_ + g)*(this->conv_in_channels_/this->group_)*ntiles_h_*ntiles_w_,
                  (float)1, weight_diff + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_));
            }
          }
          // weight_diff has (tile_h_in*tile_w_in) x (conv_out_channels) x (conv_in_channels/group) dimension
#ifdef PROFILE_WINOGRAD
          LOG(INFO) << "Convolution for weight gradient takes " << timer.MilliSeconds()/1000;
#endif
          
//          for (int i = 0; i < tile_h_in_*tile_w_in_*this->conv_out_channels_*(this->conv_in_channels_/this->group_); ++i) {
//            if (isnan(weight_diff[i])) {
//              ostringstream str;
//              str << "nan at weight_diff[" << i << "]";
//              LOG(FATAL) << str.str();
//            }
//          }

          if (false/*n == this->num_ - 1*/) {
            float *temp_weight = new float[this->conv_out_channels_*(this->conv_in_channels_/this->group_)*kernel_h*kernel_w];

            caffe_cpu_gemm<float>(CblasTrans, CblasNoTrans,
                this->conv_out_channels_*(this->conv_in_channels_/this->group_), kernel_h*kernel_w, tile_h_in_*tile_w_in_,
                (float)1, weight_diff, GKronG->get()->cpu_data(),
                (float)0, temp_weight);

            fprintf(stderr, "weight_diff[%d]\n", n);
            for (int m = 0; m < this->conv_out_channels_; ++m) {
              for (int c = 0; c < this->conv_in_channels_/this->group_; ++c) {
                for (int i = 0; i < kernel_h*kernel_w; ++i) {
                  fprintf(stderr, "%g ", temp_weight[(m*(this->conv_in_channels_/this->group_) + c)*kernel_h*kernel_w + i]);
                }
              }
              fprintf(stderr, "\n");
            }
            delete[] temp_weight;

            fprintf(stderr, "weight_diff_winograd[%d]\n", n);
            for (int n = 0; n < this->conv_out_channels_; ++n) {
              for (int c = 0; c < this->conv_in_channels_; ++c) {
                for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
                  fprintf(stderr, "%g ", weight_diff[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
                }
              }
              fprintf(stderr, "\n");
            }
          }
        }

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
#ifdef PROFILE_WINOGRAD
          timer.Start();
#endif
          // Convolution in Winograd domain
          for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
            for (int g = 0; g < this->group_; ++g) {
              caffe_cpu_gemm<float>(CblasTrans, CblasNoTrans,
                  this->conv_in_channels_/this->group_, ntiles_h_*ntiles_w_, this->conv_out_channels_/this->group_,
                  (float)1,
                  weight + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*(this->conv_in_channels_/this->group_),
                  temp1_.cpu_data() + (j*this->group_ + g)*(this->conv_out_channels_/this->group_)*ntiles_h_*ntiles_w_,
                  (float)0, col_buff + (j*this->group_ + g)*(this->conv_in_channels_/this->group_)*ntiles_h_*ntiles_w_);
            }
          }
          // col_buff has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w)
#ifdef PROFILE_WINOGRAD
          LOG(INFO) << "Convolution for bottom gradient takes " << timer.MilliSeconds()/1000;
#endif

          // Transform back to time domain
#ifdef PROFILE_WINOGRAD
          timer.Start();
#endif
          caffe_cpu_gemm<float>(CblasTrans, CblasTrans,
              this->conv_in_channels_*ntiles_h_*ntiles_w_, tile_h_in_*tile_w_in_, tile_h_in_*tile_w_in_,
              (float)1, col_buff, BKronB->get()->cpu_data(),
              (float)0, temp1_.mutable_cpu_data());
#ifdef PROFILE_WINOGRAD
          LOG(INFO) << "Inverse transformation of bottom_diff takes " << timer.MilliSeconds()/1000;
#endif

#ifdef PROFILE_WINOGRAD
          timer.Start();
#endif
          winograd_input_col2im_cpu(temp1_.cpu_data(), bottom_diff + n*this->bottom_dim_);
#ifdef PROFILE_WINOGRAD
          LOG(INFO) << "winograd_input_col2im takes " << timer.MilliSeconds()/1000;
#endif

//          for (int i = 0; i < this->bottom_dim_; ++i) {
//            if (isnan(bottom_diff[i])) {
//              ostringstream str;
//              str << "nan at bottom_diff[" << n << ", " << i << "]";
//            }
//          }
        }
      } // for each image
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WinogradLayer);
#endif

INSTANTIATE_CLASS(WinogradLayer);
REGISTER_LAYER_CLASS(Winograd);

}  // namespace caffe
