#include <vector>

#include <cfloat>
#include <omp.h>

#include "caffe/filler.hpp"
#include "caffe/solver.hpp"
#include "caffe/layers/inner_product_relu_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/csrmm.hpp"

extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

template<typename Dtype>
InnerProductReLUDropoutLayer<Dtype>::InnerProductReLUDropoutLayer(const LayerParameter& param) :
    Layer<Dtype>(param),
#ifndef CAFFE_USE_LIBXSMM_SPMDM
    weight_values_blocked_(NULL), weight_j_blocked_(NULL), weight_i_blocked_(NULL),
#endif
    bottom_transposed_(NULL)
{

}

template<typename Dtype>
InnerProductReLUDropoutLayer<Dtype>::~InnerProductReLUDropoutLayer()
{
  free(bottom_transposed_);

#ifdef CAFFE_USE_LIBXSMM_SPMDM
  libxsmm_spmdm_destroy(&libxsmm_spmdm_handle_);
#else
  free(weight_values_blocked_);
  free(weight_j_blocked_);
  free(weight_i_blocked_);
#endif
}

template<>
void InnerProductReLUDropoutLayer<double>::WeightAlign(){
  NOT_IMPLEMENTED;
}

#ifdef __AVX512F__
static int col_block_size = 256;
#else
static int col_block_size = 128;
#endif

template<>
void InnerProductReLUDropoutLayer<float>::WeightAlign(){
	const LayerParameter& layerparam = this->layer_param();
	LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity(Solver<float>::getMeasureThreshold()) << " transpose " << transpose_;

	if (layerparam.inner_product_param().dump_parameter()) {
	  this->blobs_[0]->WriteToNistMMIOSparse(layerparam.name()+".mtx");
	}

  caffe::InnerProductParameter_GemmMode gemm_mode = layerparam.inner_product_param().gemm_mode();

  if (caffe::InnerProductParameter_GemmMode_SPMDM == gemm_mode) {
    if (!transpose_) {
      if (!bias_term_) LOG(FATAL) << "SPMDM mode only works with bias term";
      if (this->layer_param_.relu_param().negative_slope() != 0) {
        LOG(FATAL) << "InnerProduct layer fused with ReLU in SPMDM mode only works with ReLU using 0 negative slop";
      }

#ifdef CAFFE_USE_LIBXSMM_SPMDM
      libxsmm_spmdm_init(N_, M_, K_, omp_get_max_threads(), &libxsmm_spmdm_handle_, &libxsmm_csr_weight_);

      int nCreateSparseSliceBlocks = libxsmm_spmdm_get_num_createSparseSlice_blocks(&libxsmm_spmdm_handle_);
#pragma omp parallel for
      for (int i = 0; i < nCreateSparseSliceBlocks; ++i) {
        libxsmm_spmdm_createSparseSlice_fp32_thread(
            &libxsmm_spmdm_handle_, 'N' /*transA*/,
            this->blobs_[0]->cpu_data(), libxsmm_csr_weight_, i, omp_get_thread_num(), omp_get_num_threads());
      }

      nnz_weight_ = 0;
      for (int i = 0; i < nCreateSparseSliceBlocks; ++i) {
        int kb = i/libxsmm_spmdm_handle_.mb;
        int mb = i%libxsmm_spmdm_handle_.mb;
        int nrows = std::min((mb + 1)*libxsmm_spmdm_handle_.bm, libxsmm_spmdm_handle_.m) - mb*libxsmm_spmdm_handle_.bm;
        nnz_weight_ += libxsmm_csr_weight_[i].rowidx[nrows];
      }

#ifndef NDEBUG
      int nnz_ref = 0;
      for (int i = 0; i < N_*K_; ++i) {
        if (this->blobs_[0]->cpu_data()[i] != 0.) ++nnz_ref;
      }
      assert(nnz_weight_ == nnz_ref);
#endif
#else
      if (M_%VLEN != 0) {
        LOG(FATAL) << "InnerProductReLUDropoutLayer in SPMDM mode requires batch size to be a multiple of " << VLEN;
      }
      int num_of_C_col_partitions = 1; // M_/(VLEN*CSRMM_REG_BLOCK_SIZE);
        // TODO: num_of_C_col_partitions is currently fixed to 1
        // To use num_of_C_col_partitions > 1, we need to rearrange output matrix after SpMDM
      if (omp_get_max_threads()%num_of_C_col_partitions != 0) {
        LOG(WARNING) << "InnerProductReLUDropoutLayer in SPMDM mode performs best when num of threads is a multiple of " << num_of_C_col_partitions;
      }
      if (omp_get_max_threads() < num_of_C_col_partitions != 0) {
        LOG(FATAL) << "InnerProductReLUDropoutLayer in SPMDM mode requires num of threads should not be less than " << num_of_C_col_partitions;
      }

      int num_of_A_col_blocks = K_/col_block_size;
      posix_memalign((void **)&weight_i_blocked_, 4096, sizeof(int)*(N_*num_of_A_col_blocks + 1));
      posix_memalign((void **)&weight_j_blocked_, 4096, sizeof(int)*K_*N_);
      posix_memalign((void **)&weight_values_blocked_, 4096, sizeof(float)*K_*N_);

      weight_i_blocked_[0] = 0;
      int nnz = 0;
      int nthreads = omp_get_max_threads()/num_of_C_col_partitions*num_of_C_col_partitions;
      int i_per_thread = (N_ + nthreads - 1)/nthreads;
      for (int tid = 0; tid < nthreads; ++tid) {
        int ibegin = std::min(i_per_thread*tid, N_);
        int iend = std::min(ibegin + i_per_thread, N_);
        for (int cb = 0; cb < num_of_A_col_blocks; ++cb) {
          for (int i = ibegin; i < iend; ++i) {
            for (int j = cb*col_block_size; j < (cb + 1)*col_block_size; ++j) {
              float v = this->blobs_[0]->mutable_cpu_data()[i*K_ + j];
              if (v != 0) {
                weight_j_blocked_[nnz] = M_*j; // pre-multiply column index with the width of dense matrix to be multipled with
                weight_values_blocked_[nnz] = v;
                ++nnz;
              }
            }
            weight_i_blocked_[num_of_A_col_blocks*ibegin + cb*(iend - ibegin) + (i - ibegin) + 1] = nnz;
          }
        }
      }
#endif
    }
    else {
      LOG(WARNING) << "SPMDM mode is not supported for transposed inner product. Falling back to GEMM mode";
    }
  }
  else if (caffe::InnerProductParameter_GemmMode_SPGEMM == gemm_mode) {
    LOG(FATAL) << "SPGEMM mode is not supported yet";
	}

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
void InnerProductReLUDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
void InnerProductReLUDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
void InnerProductReLUDropoutLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template<>
void InnerProductReLUDropoutLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  float* bottom_data = bottom[0]->mutable_cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  float* weight = this->blobs_[0]->mutable_cpu_data();

  bool PRINT_FEATURE_SPARSITY = false;
  if (PRINT_FEATURE_SPARSITY) {
    int cnt = 0;
#pragma omp parallel for reduction(+:cnt)
    for (int i = 0; i < M_*K_; ++i) {
      if (bottom_data[i] == 0) ++cnt;
    }
    LOG(INFO) << this->layer_param_.name() << " M " << M_ << " K " << K_ << " N " << N_ << " sparsity " << (double)cnt/(M_*K_);
  }

  const LayerParameter& layerparam = this->layer_param();
  caffe::InnerProductParameter_GemmMode gemm_mode = layerparam.inner_product_param().gemm_mode();

  if (caffe::InnerProductParameter_GemmMode_SPMDM == gemm_mode && !transpose_) {
#ifdef CAFFE_USE_LIBXSMM_SPMDM
    double t = omp_get_wtime();

    int num_compute_blocks = libxsmm_spmdm_get_num_compute_blocks(&libxsmm_spmdm_handle_);
#pragma omp parallel for
    for (int i = 0; i < num_compute_blocks; ++i) {
      float alpha = 1, beta = 0;
      libxsmm_spmdm_compute_fp32_thread(
          &libxsmm_spmdm_handle_, 'N' /*transA*/, layerparam.inner_product_param().spmdm_transpose_in() ? 'T' : 'N' /*transB*/,
          &alpha, libxsmm_csr_weight_,
          bottom_data,
          layerparam.inner_product_param().spmdm_transpose_out() ? 'T' : 'N' /*transC*/, &beta, top_data,
          i, omp_get_thread_num(), omp_get_num_threads());
    }

    t = omp_get_wtime() - t;
    LOG(INFO) << "csrmm takes " << t << " effective GF/s " << 2.*K_*N_*M_/t/1e9 << " real GF/s " << 2.*nnz_weight_*M_/t/1e9;

    if (bias_term_) {
      // JSP: common path for AlexNet
      caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
          bias_multiplier_.cpu_data(),
          this->blobs_[1]->cpu_data(), (float)1., top_data);
    }

    const int count = top[0]->count();
    float negative_slope = this->layer_param_.relu_param().negative_slope();
    if (0 == negative_slope) {
#pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(top_data[i], float(0));
      }
    }
    else {
#pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(top_data[i], float(0))
            + negative_slope * std::min(top_data[i], float(0));
      }
    }
#else
    if (layerparam.inner_product_param().spmdm_transpose_in()) {
      mkl_somatcopy('R', 'T', M_, K_, 1, bottom_data, K_, bottom_transposed_, M_);
    }

    int num_of_A_col_blocks = K_/col_block_size;
    int num_of_C_col_partitions = 1; //M_/(VLEN*CSRMM_REG_BLOCK_SIZE);
    double t = omp_get_wtime();
    csrmm_fused_C_decomposed(
        weight_values_blocked_, weight_j_blocked_, weight_i_blocked_,
        layerparam.inner_product_param().spmdm_transpose_in() ? bottom_transposed_ : bottom_data,
        top_data,
        N_, M_, K_,
        this->blobs_[1]->cpu_data(),
        num_of_C_col_partitions,
        num_of_A_col_blocks);
    t = omp_get_wtime() - t;
    LOG(INFO) << "csrmm takes " << t << " effective GF/s " << 2.*K_*N_*M_/t/1e9 << " real GF/s " << 2.*weight_i_blocked_[num_of_A_col_blocks*N_]*M_/t/1e9;

    if (layerparam.inner_product_param().spmdm_transpose_out()) {
      memcpy(bottom_transposed_, top_data, sizeof(float)*M_*N_);
      mkl_somatcopy('R', 'T', N_, M_, 1, bottom_transposed_, M_, top_data, N_);
    }
#endif

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
  else
  {
    caffe_cpu_gemm<float>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
        M_, N_, K_, (float)1.,
        bottom_data, weight, (float)0., top_data);
    if (bias_term_) {
      // JSP: common path for AlexNet
      caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
          bias_multiplier_.cpu_data(),
          this->blobs_[1]->cpu_data(), (float)1., top_data);
    }

    const int count = top[0]->count();
    float negative_slope = this->layer_param_.relu_param().negative_slope();
    if (0 == negative_slope) {
#pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(top_data[i], float(0));
      }
    }
    else {
#pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(top_data[i], float(0))
            + negative_slope * std::min(top_data[i], float(0));
      }
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
void InnerProductReLUDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(InnerProductReLUDropoutLayer);
#else
template <typename Dtype>
void InnerProductReLUDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void InnerProductReLUDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(InnerProductReLUDropoutLayer);
REGISTER_LAYER_CLASS(InnerProductReLUDropout);

}  // namespace caffe
