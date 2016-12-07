#include <vector>

#include "caffe/layers/fft_layer.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

template <typename Dtype>
void FFTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void FFTLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(FFTLayer);

}  // namespace caffe
