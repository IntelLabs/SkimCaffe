#include <vector>

#include "caffe/filler.hpp"
#include "caffe/solver.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/spgemm.hpp"

extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

template<typename Dtype>
InnerProductLayer<Dtype>::InnerProductLayer(const LayerParameter& param) :
    Layer<Dtype>(param),
    bottom_values_(NULL), bottom_j_(NULL), bottom_i_(NULL),
    top_values_(NULL), top_j_(NULL), top_i_(NULL),
    weight_values_(NULL), weight_j_(NULL), weight_i_(NULL),
    bottom_transposed_(NULL),
    weight_values_blocked_(NULL), weight_j_blocked_(NULL), weight_i_blocked_(NULL)
{

}

template<typename Dtype>
InnerProductLayer<Dtype>::~InnerProductLayer()
{
  free(bottom_values_);
  free(bottom_j_);
  free(bottom_i_);
  free(bottom_transposed_);

  free(top_values_);
  free(top_j_);
  free(top_i_);

  free(weight_values_);
  free(weight_j_);
  free(weight_i_);

  free(weight_values_blocked_);
  free(weight_j_blocked_);
  free(weight_i_blocked_);
}

template<>
void InnerProductLayer<double>::WeightAlign(){
  NOT_IMPLEMENTED;
}

static int col_block_size = 128;

template<>
void InnerProductLayer<float>::WeightAlign(){
	const LayerParameter& layerparam = this->layer_param();
	LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity(Solver<float>::getMeasureThreshold()) << " transpose " << transpose_;

	if (layerparam.inner_product_param().dump_parameter()) {
	  this->blobs_[0]->WriteToNistMMIOSparse(layerparam.name()+".mtx");
	}

	posix_memalign((void **)&weight_i_, 4096, sizeof(int)*(std::max(K_, N_) + 1));
	posix_memalign((void **)&weight_j_, 4096, sizeof(int)*K_*N_);
	posix_memalign((void **)&weight_values_, 4096, sizeof(float)*K_*N_);

	caffe::InnerProductParameter_GemmMode gemm_mode = layerparam.inner_product_param().gemm_mode();

  if (caffe::InnerProductParameter_GemmMode_SPMDM == gemm_mode) {
    if (!transpose_) {
      int ncolblocks = K_/col_block_size;
      posix_memalign((void **)&weight_i_blocked_, 4096, sizeof(int)*(N_*ncolblocks + 1));
      posix_memalign((void **)&weight_j_blocked_, 4096, sizeof(int)*K_*N_);
      posix_memalign((void **)&weight_values_blocked_, 4096, sizeof(float)*K_*N_);

      weight_i_blocked_[0] = 0;
      int nnz = 0;
      for (int cb = 0; cb < ncolblocks; ++cb) {
        for (int i = 0; i < N_; ++i) {
          for (int j = cb*col_block_size; j < (cb + 1)*col_block_size; ++j) {
            float v = this->blobs_[0]->mutable_cpu_data()[i*K_ + j];
            if (v != 0) {
              weight_j_blocked_[nnz] = M_*j;
              weight_values_blocked_[nnz] = v;
              ++nnz;
            }
          }
          weight_i_blocked_[cb*N_ + i + 1] = nnz;
        }
      }
    }
    else {
      LOG(WARNING) << "SPMDM mode is not supported for transposed inner product. Falling back to GEMM mode";
    }
  }
  else if (caffe::InnerProductParameter_GemmMode_SPGEMM == gemm_mode) {
    LOG(FATAL) << "SPGEMM mode is not supported yet";
  }

  posix_memalign((void **)&bottom_i_, 4096, sizeof(int)*(std::max(M_, K_) + 1));
  posix_memalign((void **)&bottom_j_, 4096, sizeof(int)*M_*K_);
  posix_memalign((void **)&bottom_values_, 4096, sizeof(float)*M_*K_);

  posix_memalign((void **)&top_i_, 4096, sizeof(int)*(std::max(M_, N_) + 1));
  posix_memalign((void **)&top_j_, 4096, sizeof(int)*M_*N_);
  posix_memalign((void **)&top_values_, 4096, sizeof(float)*M_*N_);

  posix_memalign((void **)&bottom_transposed_, 4096, sizeof(int)*M_*std::max(K_, N_));

	//disconnect connections
	if( layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_ELTWISE ){
		LOG(INFO)<<"all zero weights of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<float>::ELTWISE, Solver<float>::getPruneThreshold());
	}else if(layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_GRPWISE){
		LOG(INFO)<<"weights lying in all-zero groups of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<float>::GRPWISE, Solver<float>::getPruneThreshold());
	}
}

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
    // if true, weight is in row-major, otherwise it's in col-major
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<>
void InnerProductLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template<>
void InnerProductLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  float* bottom_data = bottom[0]->mutable_cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  float* weight = this->blobs_[0]->mutable_cpu_data();

  bool PRINT_FEATURE_SPARSITY = false;
  if (PRINT_FEATURE_SPARSITY) {
    int cnt = 0;
    for (int i = 0; i < M_*K_; ++i) {
      if (bottom_data[i] == 0) ++cnt;
    }
    LOG(INFO) << this->layer_param_.name() << " M " << M_ << " K " << K_ << " N " << N_ << " sparsity " << (double)cnt/(M_*K_);
  }

  const LayerParameter& layerparam = this->layer_param();
  caffe::InnerProductParameter_GemmMode gemm_mode = layerparam.inner_product_param().gemm_mode();

  if (caffe::InnerProductParameter_GemmMode_SPMDM == gemm_mode && !transpose_) {
    if (layerparam.inner_product_param().spmdm_transpose_in()) {
      mkl_somatcopy('R', 'T', M_, K_, 1, bottom_data, K_, bottom_transposed_, M_);
    }

    int ncolblocks = K_/col_block_size;
    double t = omp_get_wtime();
    csrmm(
        weight_values_blocked_, weight_j_blocked_, weight_i_blocked_,
        layerparam.inner_product_param().spmdm_transpose_in() ? bottom_transposed_ : bottom_data,
        top_data,
        N_, M_, K_,
        this->blobs_[1]->cpu_data(),
        col_block_size);
    t = omp_get_wtime() - t;
    LOG(INFO) << "csrmm takes " << t << " effective GF/s " << 2.*K_*N_*M_/t/1e9 << " real GF/s " << 2.*weight_i_blocked_[ncolblocks*N_]*M_/t/1e9;

    if (layerparam.inner_product_param().spmdm_transpose_out()) {
      memcpy(bottom_transposed_, top_data, sizeof(float)*M_*N_);
      mkl_somatcopy('R', 'T', N_, M_, 1, bottom_transposed_, M_, top_data, N_);
    }

    std::string name(this->layer_param_.name());
    if (total_conv_cycles.find(name) == total_conv_cycles.end()) {
      total_conv_cycles[name] = 0;
      total_conv_flops[name] = 0;
    }
    total_conv_cycles[name] += t*get_cpu_freq();
    total_conv_flops[name] += 2.*M_*K_*N_;
    total_files += M_;
  }
  else if (caffe::InnerProductParameter_GemmMode_SPGEMM == gemm_mode) {
    LOG(FATAL) << "SPGEMM mode is not supported yet";
  }
  else {
    // activation_matrix * weight_matrix^T
    caffe_cpu_gemm<float>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
        M_, N_, K_, (float)1.,
        bottom_data, weight, (float)0., top_data);

    if (bias_term_) {
      // JSP: common path for AlexNet
      caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
          bias_multiplier_.cpu_data(),
          this->blobs_[1]->cpu_data(), (float)1., top_data);
    }
  }

  if (layerparam.inner_product_param().dump_activation()) {
    static std::map<std::string, int> mtx_cnt_map;
    if (mtx_cnt_map.find(layerparam.name()) == mtx_cnt_map.end()) {
      mtx_cnt_map[layerparam.name()] = 0;
    }

    char mtx_name[1024];
    sprintf(mtx_name, "%s_in_%d.mtx", layerparam.name().c_str(), mtx_cnt_map[layerparam.name()]);
    bottom[0]->WriteToNistMMIOSparse(mtx_name);

    sprintf(mtx_name, "%s_out_%d.mtx", layerparam.name().c_str(), mtx_cnt_map[layerparam.name()]);
    top[0]->WriteToNistMMIOSparse(mtx_name);

    ++mtx_cnt_map[layerparam.name()];
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
