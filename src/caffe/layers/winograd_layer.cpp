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
void WinogradLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];
  WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);

  const int tile_h_in = GKronG->M;
  const int tile_w_in = GKronG->M;
  const int tile_h_out = tile_h_in - GKronG->N + 1, tile_w_out = tile_w_in - GKronG->N + 1;

  int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];

  int ntiles_h = (height + kernel_h - 1 + tile_h_out - 1)/tile_h_out;
  int ntiles_w = (width + kernel_w - 1 + tile_w_out - 1)/tile_w_out;

  // temp1_, temp2_, and winograd_ are stored in transposed form
  vector<int> shape;
  shape.push_back(tile_h_in*tile_w_in);
  shape.push_back(this->conv_in_channels_/this->group_);
  shape.push_back(ntiles_h*ntiles_w);
  temp1_.Reshape(shape);

  shape.clear();
  shape.push_back(tile_h_in*tile_w_in);
  shape.push_back(this->conv_out_channels_);
  shape.push_back(ntiles_h*ntiles_w);
  temp2_.Reshape(shape);

  shape.clear();
  shape.push_back(this->conv_out_channels_);
  shape.push_back(ntiles_h*ntiles_w);
  shape.push_back(tile_h_out*tile_w_out);
  output_temp_.Reshape(shape);

  shape.clear();
  shape.push_back(tile_h_in*tile_w_in);
  shape.push_back(this->conv_out_channels_);
  shape.push_back(this->conv_in_channels_/this->group_);
  winograd_weight_.Reshape(shape);
}

template<typename Dtype>
void WinogradLayer<Dtype>::winograd_im2col_cpu(const Dtype *data, Dtype *col_buff)
{
  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];
  WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);

  const int tile_h_in = GKronG->M;
  const int tile_w_in = GKronG->M;
  const int tile_h_out = tile_h_in - GKronG->N + 1, tile_w_out = tile_w_in - GKronG->N + 1;

  int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];

  int ntiles_h = (height + kernel_h - 1 + tile_h_out - 1)/tile_h_out;
  int ntiles_w = (width + kernel_w - 1 + tile_w_out - 1)/tile_w_out;

  int pad_h = this->pad_.cpu_data()[0];
  int pad_w = this->pad_.cpu_data()[1];

  for (int c = 0; c < this->conv_in_channels_; ++c) {
    for (int tile_h = 0; tile_h < ntiles_h; ++tile_h) {
      for (int tile_w = 0; tile_w < ntiles_w; ++tile_w) {
        for (int y = 0; y < tile_h_in; ++y) {
          for (int x = 0; x < tile_w_in; ++x) {
            int in_y = tile_h*tile_h_out + y - pad_h;
            int in_x = tile_w*tile_w_out + x - pad_w;

            if (in_y < 0 || in_x < 0 || in_y >= height || in_x >= width) {
              col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x] = 0;
            }
            else {
              col_buff[(((c*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x] =
                  data[(c*height + in_y)*width + in_x];
            }
          }
        }
      } // for each tile
    } // for each tile
  } // for each input channel
}

template<typename Dtype>
void WinogradLayer<Dtype>::winograd_col2im_cpu(const Dtype *col_buff, Dtype *data)
{
  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];
  WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);

  const int tile_h_in = GKronG->M;
  const int tile_w_in = GKronG->M;
  const int tile_h_out = tile_h_in - GKronG->N + 1, tile_w_out = tile_w_in - GKronG->N + 1;

  int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];

  int ntiles_h = (height + kernel_h - 1 + tile_h_out - 1)/tile_h_out;
  int ntiles_w = (width + kernel_w - 1 + tile_w_out - 1)/tile_w_out;

  int pad_h = this->pad_.cpu_data()[0];
  int pad_w = this->pad_.cpu_data()[1];
  int stride_h = this->stride_.cpu_data()[0];
  int stride_w = this->stride_.cpu_data()[1];
  int dilation_h = this->dilation_.cpu_data()[0];
  int dilation_w = this->dilation_.cpu_data()[1];

  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  for (int o = 0; o < this->conv_out_channels_; ++o) {
    for (int tile_h = 0; tile_h < ntiles_h; ++tile_h) {
      for (int tile_w = 0; tile_w < ntiles_w; ++tile_w) {
        for (int y = 0; y < tile_h_out; ++y) {
          for (int x = 0; x < tile_w_out; ++x) {
            int out_y = tile_h*tile_h_out + y;
            int out_x = tile_w*tile_w_out + x;

            if (out_y < output_h && out_x < output_w) {
              data[(o*output_h + out_y)*output_w + out_x] =
                  col_buff[(((o*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x];
            }
          }
        }
      } // for each tile
    } // for each tile
  } // for each input channel
}

template <typename Dtype>
void WinogradLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
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

      WinogradBKronB<Dtype> *BKronB = WinogradBKronB<Dtype>::getInstance(kernel_h);

      const int tile_h_in = BKronB->M;
      const int tile_w_in = BKronB->M;
      const int tile_h_out = tile_h_in - BKronB->N + 1, tile_w_out = tile_w_in - BKronB->N + 1;

      int ntiles_h = (height + kernel_h - 1 + tile_h_out - 1)/tile_h_out;
      int ntiles_w = (width + kernel_w - 1 + tile_w_out - 1)/tile_w_out;

      int M = this->conv_in_channels_*ntiles_h*ntiles_w;

      int offset = n*M*tile_h_in*tile_w_in;
      Dtype *col_buff = this->col_buffer_.mutable_cpu_data() + offset;

      winograd_im2col_cpu(bottom_data + n*this->bottom_dim_, col_buff);

      // Transform input to Winograd domain
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, tile_h_in*tile_w_in, M, tile_h_in*tile_w_in,
          (Dtype)1, BKronB->get()->cpu_data(), col_buff,
          (Dtype)0, temp1_.mutable_cpu_data());
      // temp1_ has (tile_h_in*tile_w_in) x (conv_in_channels) x (ntiles_h*ntiles_w) dimension

      // Transform weight to Winograd domain
      WinogradGKronG<Dtype> *GKronG = WinogradGKronG<Dtype>::getInstance(kernel_h);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, tile_h_in*tile_w_in, (this->conv_in_channels_/this->group_)*this->conv_out_channels_, kernel_h*kernel_w,
          (Dtype)1, GKronG->get()->cpu_data(), weight,
          (Dtype)0, winograd_weight_.mutable_cpu_data());
      // wingrad_weight_ has (tile_h_in*tile_w_in) x (conv_out_channels) x (conv_in_channels) dimension

      // Convolution in Winograd domain
      for (int j = 0; j < tile_h_in*tile_w_in; ++j) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_, ntiles_h*ntiles_w, this->conv_in_channels_/this->group_,
            (Dtype)1, winograd_weight_.cpu_data() + j*this->conv_out_channels_*(this->conv_in_channels_/this->group_), temp1_.cpu_data() + j*(this->conv_in_channels_/this->group_)*ntiles_h*ntiles_w,
            (Dtype)0, temp2_.mutable_cpu_data() + j*this->conv_out_channels_*ntiles_h*ntiles_w);
      }
      // temp2_ has (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w)

      // Transform back to time domain
      WinogradAKronA<Dtype> *AKronA = WinogradAKronA<Dtype>::getInstance(kernel_h);
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->conv_out_channels_*ntiles_h*ntiles_w, tile_h_out*tile_w_out, tile_h_in*tile_w_in,
          (Dtype)1, temp2_.cpu_data(), AKronA->get()->cpu_data(),
          (Dtype)0, output_temp_.mutable_cpu_data());

      winograd_col2im_cpu(output_temp_.cpu_data(), top_data + n*this->top_dim_);

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void WinogradLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
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
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WinogradLayer);
#endif

INSTANTIATE_CLASS(WinogradLayer);
REGISTER_LAYER_CLASS(Winograd);

}  // namespace caffe
