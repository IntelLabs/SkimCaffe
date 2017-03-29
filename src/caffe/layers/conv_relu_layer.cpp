#include <vector>
#include <omp.h>

#include "caffe/layers/conv_relu_layer.hpp"
#include "caffe/util/math_functions_intel.hpp"
#include "caffe/util/cpu_info.hpp"
#include "caffe/util/sconv.hpp"

extern unsigned long long conv_cycles_of_this_batch[1024*16];
extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

extern double padding_time;

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::compute_output_shape() {
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
void ConvolutionReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    conv_cycles_of_this_batch[i*16] = 0;
  }

  int height = this->conv_input_shape_.cpu_data()[1];
  int width = this->conv_input_shape_.cpu_data()[2];
  int pad_h = this->pad_.cpu_data()[0];
  int pad_w = this->pad_.cpu_data()[1];
  int kernel_h = this->kernel_shape_.cpu_data()[0];
  int kernel_w = this->kernel_shape_.cpu_data()[1];
  int stride_h = this->stride_.cpu_data()[0];
  int stride_w = this->stride_.cpu_data()[1];
  int dilation_h = this->dilation_.cpu_data()[0];
  int dilation_w = this->dilation_.cpu_data()[1];

  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
  }
  double t = omp_get_wtime();
  double t2 = 0, t3 = 0;
  padding_time = 0;
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  if (negative_slope != 0) {
    LOG(FATAL) << type() << " only supports negative_slope == 0";
  }

  // JSP: by some reason, if nested omp parallelism is used for MKL, I get a wrong results.
  // Disable nested omp parallelization for now. We don't need nested parallelism as long as
  // batch size is big enough. Still, need more investigation.
  int mkl_max_threads_saved = mkl_get_max_threads();
  mkl_set_num_threads(1);

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

//  int nnz_input = 0;
//  int num_of_non_zero_channels = 0;
//  for (int n = 0; n < this->num_; ++n) {
//    for (int ic = 0; ic < this->conv_in_channels_; ++ic) {
//      bool is_non_zero_channel = true;
//      for (int h = 0; h < height; ++h) {
//        for (int w = 0; w < width; ++w) {
//          if (bottom_data[((n*this->conv_in_channels_ + ic)*height + h)*width + w] == 0) ++nnz_input;
//          else is_non_zero_channel = false;
//        }
//      }
//      if (is_non_zero_channel) ++num_of_non_zero_channels;
//    }
//  }
//  LOG(INFO) << "element-sparsity " << (double)nnz_input/(this->num_*this->conv_in_channels_*height*width) << " channel-sparsity " << (double)num_of_non_zero_channels/(this->num_*this->conv_in_channels_);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      int n_begin, n_end;
      cpu::OpenMpManager::getBatchThreadPartition(&n_begin, &n_end, this->num_);

      for (int n = n_begin; n < n_end; ++n) { // JSP: this->num_ is batch size
        Dtype *top_current = top_data + n * this->top_dim_;

        if (0 == tid) t2 -= omp_get_wtime();
        this->forward_cpu_gemm(
              bottom_data + n * this->bottom_dim_, weight, top_current, n);

        if (0 == tid) t2 += omp_get_wtime();
        if (this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
          if (this->bias_term_) {
            // JSP: common path of AlexNet
            this->forward_cpu_bias(top_current, bias);
          }
          // bias is not fused when conv mode is not DIRECT_SCONV
          for (int i = 0; i < this->top_dim_; ++i) {
            top_current[i] = std::max(top_current[i], Dtype(0));
          }
        }
      } // for each input in the batch
    }
  }

  mkl_set_num_threads(mkl_max_threads_saved);

  LOG(INFO) << this->layer_param_.name() << " wall clock-time " << omp_get_wtime() - t << " padding-time " << padding_time;

  double flops = (double)this->num_*this->conv_out_channels_*this->conv_in_channels_/this->group_*output_h*output_w*kernel_h*kernel_w*2;

  unsigned long long max_conv_cycle = 0, sum_conv_cycle = 0;
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    max_conv_cycle = std::max(max_conv_cycle, conv_cycles_of_this_batch[i*16]);
    sum_conv_cycle += conv_cycles_of_this_batch[i*16];
  }
  std::string name(this->layer_param_.name());
  LOG(INFO) <<
      name <<
      " K-cycles-per-file max " << max_conv_cycle/1000./this->num_ <<
      " avg " << sum_conv_cycle/1000./omp_get_max_threads()/this->num_ <<
      " mFlops-per-file " << flops/this->num_/1e6 <<
      " GF/s " << flops/(max_conv_cycle/get_cpu_freq())/1e9;

  if (total_conv_cycles.find(name) == total_conv_cycles.end()) {
    total_conv_cycles[name] = 0;
    total_conv_flops[name] = 0;
  }
  total_conv_cycles[name] += max_conv_cycle;
  total_conv_flops[name] += flops;
  total_files += this->num_;
}

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionReLULayer);
#else
template <typename Dtype>
void ConvolutionReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(ConvolutionReLULayer);
REGISTER_LAYER_CLASS(ConvolutionReLU);

}  // namespace caffe
