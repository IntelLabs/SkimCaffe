#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/winograd.hpp"
#include <omp.h>

extern unsigned long long conv_cycles_of_this_batch[1024*16];
extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

extern double padding_time;

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
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
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int height = this->conv_input_shape_.cpu_data()[1];
  int width = this->conv_input_shape_.cpu_data()[2];
  int kernel_h = this->kernel_shape_.cpu_data()[0];
  int kernel_w = this->kernel_shape_.cpu_data()[1];
  int pad_h = this->pad_.cpu_data()[0];
  int pad_w = this->pad_.cpu_data()[1];
  int stride_h = this->stride_.cpu_data()[0];
  int stride_w = this->stride_.cpu_data()[1];
  int dilation_h = this->dilation_.cpu_data()[0];
  int dilation_w = this->dilation_.cpu_data()[1];

  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
  }
  double t = omp_get_wtime();

  // JSP: by some reason, if nested omp parallelism is used for MKL, I get a wrong results.
  // Disable nested omp parallelization for now. We don't need nested parallelism as long as
  // batch size is big enough. Still, need more investigation.
  int mkl_max_threads_saved = mkl_get_max_threads();
  mkl_set_num_threads(1);

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
#pragma omp parallel for
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_, n);
      if (this->bias_term_) {
        // bias term is fused with direct convolution for second layer
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }

    const LayerParameter& layerparam = this->layer_param();

    if (layerparam.convolution_param().dump_activation()) {
      assert(bottom.size() == 1);
      static std::map<std::string, int> mtx_cnt_map;
      if (mtx_cnt_map.find(layerparam.name()) == mtx_cnt_map.end()) {
        mtx_cnt_map[layerparam.name()] = 0;
      }

      for (int n = 0; n < this->num_; ++n) {
        char mtx_name[1024];
        sprintf(mtx_name, "%s_in_%d_%d.mtx", layerparam.name().c_str(), mtx_cnt_map[layerparam.name()], n);
        if (bottom[i]->num_axes() != 4) {
          printf("%s\n", bottom[i]->shape_string().c_str());
        }
        assert(bottom[i]->num_axes() == 4);
        Blob<Dtype>::Write3DTensorToNistMMIOSparse(
            mtx_name, bottom_data + n * this->bottom_dim_,
            bottom[i]->shape(1), bottom[i]->shape(2), bottom[i]->shape(3));

        sprintf(mtx_name, "%s_out_%d_%d.mtx", layerparam.name().c_str(), mtx_cnt_map[layerparam.name()], n);
        assert(top[i]->num_axes() == 4);
        Blob<Dtype>::Write3DTensorToNistMMIOSparse(
            mtx_name, top_data + n * this->top_dim_,
            top[i]->shape(1), top[i]->shape(2), top[i]->shape(3));
      }

      ++mtx_cnt_map[layerparam.name()];
    }
  }

  mkl_set_num_threads(mkl_max_threads_saved);

  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

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
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
