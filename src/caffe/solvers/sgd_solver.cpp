#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/winograd.hpp"

namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  temp_winograd_.clear();
  unthresholded_.clear();
  temp_winograd_transform_.clear();
  temp_winograd_transform_ptrs_.clear();
  temp_winograd_weight_.clear();
  temp_winograd_weight_ptrs_.clear();

  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    if (shape.size() == 4 && (shape[2] == 3 || shape[2] == 5) && shape[2] == shape[3]) {

      const vector<string>& net_params_local_regular_types = this->net_->params_regularization_type();
      string regularization_type = this->param_.regularization_type();
      string local_regularization_type = net_params_local_regular_types[i];
      if(!local_regularization_type.empty()){
        regularization_type = local_regularization_type;
      }

      if (regularization_type == "L1_DNS" || regularization_type == "L1_DNS_Winograd") {
        unthresholded_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
      }
      else {
        unthresholded_.push_back(shared_ptr<Blob<Dtype> >((Blob<Dtype> *)NULL));
      }

      if (regularization_type == "L1_Winograd" || regularization_type == "L2_Winograd") {
        int N = shape[2];
        int M = N == 3 ? 6 : 8;

        vector<int> shape_winograd;
        shape_winograd.push_back(shape[0]);
        shape_winograd.push_back(shape[1]);
        shape_winograd.push_back(M);
        shape_winograd.push_back(M);
        temp_winograd_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape_winograd)));

        vector<int> shape_winograd_weight;
        shape_winograd_weight.push_back(shape[0]);
        shape_winograd_weight.push_back(shape[1]);
        shape_winograd_weight.push_back(M*M + N*N);
        temp_winograd_weight_.push_back(shared_ptr<Blob<double> >(new Blob<double>(shape_winograd_weight)));

        vector<int> shape_ptr;
        shape_ptr.push_back(shape[0]);
        shape_ptr.push_back(shape[1]);
        temp_winograd_weight_ptrs_.push_back(shared_ptr<Blob<long> >(new Blob<long>(shape_ptr)));

        vector<int> shape_winograd_transform;
        shape_winograd_transform.push_back(shape[0]);
        shape_winograd_transform.push_back(shape[1]);
        shape_winograd_transform.push_back(N*N);
        shape_winograd_transform.push_back(M*M + N*N);
        temp_winograd_transform_.push_back(shared_ptr<Blob<double> >(new Blob<double>(shape_winograd_transform)));

        temp_winograd_transform_ptrs_.push_back(shared_ptr<Blob<long> >(new Blob<long>(shape_ptr)));

        switch (Caffe::mode()) {
        case Caffe::GPU:
        {
#ifndef CPU_ONLY
          double *winograd_transform_cpu = temp_winograd_transform_.back()->mutable_cpu_data();
          double *winograd_weight_cpu = temp_winograd_weight_.back()->mutable_cpu_data();

          for (int i = 0; i < shape[0]*shape[1]; ++i) {
            for (int j = 0; j < M*M; ++j) {
              winograd_weight_cpu[i*(M*M + N*N) + j] = 0;
            }
            // winograd_transform should be stored in column major for cublasDgelsBatched
            for (int j = M*M; j < M*M + N*N; ++j) {
              for (int k = 0; k < N*N; ++k) {
                winograd_transform_cpu[(i*(N*N) + k)*(M*M + N*N) + j] = 0;
              }
              winograd_transform_cpu[(i*(N*N) + j - M*M)*(M*M + N*N) + j] = 1;
            }
          }

          double *winograd_transform = temp_winograd_transform_.back()->mutable_gpu_data();
          double **winograd_transform_ptrs = (double **)temp_winograd_transform_ptrs_.back()->mutable_cpu_data();

          double *winograd_weight = temp_winograd_weight_.back()->mutable_gpu_data();
          double **winograd_weight_ptrs = (double **)temp_winograd_weight_ptrs_.back()->mutable_cpu_data();

          for (int i = 0; i < shape[0]*shape[1]; ++i) {
            winograd_transform_ptrs[i] = winograd_transform + i*(M*M + N*N)*(N*N);
            winograd_weight_ptrs[i] = winograd_weight + i*(M*M + N*N);
          }
#else
          NO_GPU;
#endif
          break;
        }
        }
      }
      else {
        temp_winograd_.push_back(shared_ptr<Blob<Dtype> >((Blob<Dtype> *)NULL));
        temp_winograd_weight_.push_back(shared_ptr<Blob<double> >((Blob<double> *)NULL));
        temp_winograd_weight_ptrs_.push_back(shared_ptr<Blob<long> >((Blob<long> *)NULL));
        temp_winograd_transform_.push_back(shared_ptr<Blob<double> >((Blob<double> *)NULL));
        temp_winograd_transform_ptrs_.push_back(shared_ptr<Blob<long> >((Blob<long> *)NULL));
      }
    }
    else {
      unthresholded_.push_back(shared_ptr<Blob<Dtype> >((Blob<Dtype> *)NULL));
      temp_winograd_.push_back(shared_ptr<Blob<Dtype> >((Blob<Dtype> *)NULL));
      temp_winograd_weight_.push_back(shared_ptr<Blob<double> >((Blob<double> *)NULL));
      temp_winograd_weight_ptrs_.push_back(shared_ptr<Blob<long> >((Blob<long> *)NULL));
      temp_winograd_transform_.push_back(shared_ptr<Blob<double> >((Blob<double> *)NULL));
      temp_winograd_transform_ptrs_.push_back(shared_ptr<Blob<long> >((Blob<long> *)NULL));
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::checkIfLearnableParameterResized() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    if (history_[i]->shape() != shape) {
      history_[i]->Reshape(shape);
    }
    if (update_[i]->shape() != shape) {
      update_[i]->Reshape(shape);
    }
    if (temp_[i]->shape() != shape) {
      temp_[i]->Reshape(shape);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;

    //display sparsity
	//const vector<float>& net_params_weight_decay =
	//	  this->net_->params_weight_decay();
	//Dtype weight_decay = this->param_.weight_decay();
	ostringstream sparsity_msg_stream;
	sparsity_msg_stream << "    Element Sparsity %: \n";
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
		//Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
    if (this->net_->learnable_params()[param_id]->num_axes() >= 2) {
		  sparsity_msg_stream << GetSparsity(param_id) <<"\t";
    }
		//if(local_decay) sparsity_msg_stream << GetSparsity(param_id) <<"\t";
		//else sparsity_msg_stream << -1 <<"\t";
	}
	LOG(INFO) << sparsity_msg_stream.str();

	sparsity_msg_stream.str("");
	sparsity_msg_stream << "     Winograd Sparsity %: \n";
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() >= 2) {
		  sparsity_msg_stream << GetWinogradSparsity(param_id) <<"\t";
    }
	}
	LOG(INFO) << sparsity_msg_stream.str();

	sparsity_msg_stream.str("");
	sparsity_msg_stream << "     Winograd Old Sparsity %: \n";
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() >= 2) {
		  sparsity_msg_stream << GetWinogradSparsityOld(param_id) <<"\t";
    }
	}
	LOG(INFO) << sparsity_msg_stream.str();

  PrintWinogradFiberSliceSparsity();

#if 0
	sparsity_msg_stream.str("");
	sparsity_msg_stream << "     Column Sparsity %: \n";
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
		//Dtype local_decay = this->param_.kernel_shape_decay() * this->net_->params_kernel_shape_decay()[param_id];
    if (this->net_->learnable_params()[param_id]->num_axes() >= 2) {
		  sparsity_msg_stream << GetGroupSparsity(param_id, true) <<"\t";
    }
		//if(local_decay) sparsity_msg_stream << GetGroupSparsity(param_id, true) <<"\t";
		//else sparsity_msg_stream << -1 <<"\t";
	}
	LOG(INFO) << sparsity_msg_stream.str();

	sparsity_msg_stream.str("");
	sparsity_msg_stream << "        Row Sparsity %: \n";
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() >= 2) {
		  sparsity_msg_stream << GetGroupSparsity(param_id, false) <<"\t";
    }
		//if(local_decay) sparsity_msg_stream << GetGroupSparsity(param_id, false) <<"\t";
		//else sparsity_msg_stream << -1 <<"\t";
	}
	LOG(INFO) << sparsity_msg_stream.str();

	sparsity_msg_stream.str("");
	sparsity_msg_stream << "      Block Sparsity %: \n";
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() >= 2) {
      const vector<BlockGroupLassoSpec> net_params_block_group_lasso =
                 this->net_->params_block_group_lasso()[param_id];
      for (int blk_idx=0;blk_idx<net_params_block_group_lasso.size();blk_idx++){
        int xdimen = net_params_block_group_lasso[blk_idx].xdimen();
        int ydimen = net_params_block_group_lasso[blk_idx].ydimen();
        sparsity_msg_stream << "("<<xdimen<<","<<ydimen<<"):"<<GetGroupSparsity(param_id, ydimen, xdimen) <<";";
      }
      sparsity_msg_stream << "\t";
    }
	}
	LOG(INFO) << sparsity_msg_stream.str();
#endif

  sparsity_msg_stream.str("");
  sparsity_msg_stream << "        OC-fiber Sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() == 4) {
      sparsity_msg_stream << GetFiberSparsity(param_id, 0) <<"\t";
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();

  sparsity_msg_stream.str("");
  sparsity_msg_stream << "        IC-fiber Sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() == 4) {
      sparsity_msg_stream << GetFiberSparsity(param_id, 1) <<"\t";
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();

  sparsity_msg_stream.str("");
  sparsity_msg_stream << "        kernel-fiber Sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() == 4) {
      sparsity_msg_stream << GetFiberSparsity(param_id, 2) <<"\t";
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();

  sparsity_msg_stream.str("");
  sparsity_msg_stream << "        OC-slice Sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() == 4) {
      sparsity_msg_stream << GetSliceSparsity(param_id, 0) <<"\t";
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();

  sparsity_msg_stream.str("");
  sparsity_msg_stream << "        IC-slice Sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() == 4) {
      sparsity_msg_stream << GetSliceSparsity(param_id, 1) <<"\t";
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();

  sparsity_msg_stream.str("");
  sparsity_msg_stream << "        kernel-slice Sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    if (this->net_->learnable_params()[param_id]->num_axes() == 4) {
      sparsity_msg_stream << GetSliceSparsity(param_id, 2) <<"\t";
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();
  }

  ClipGradients();
  Solver<Dtype>::total_regularization_term_ = Dtype(0);
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);
    Solver<Dtype>::total_regularization_term_ += Regularize(param_id);
    //Solver<Dtype>::total_regularization_term_ += GroupLassoRegularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  //this->net_->Update();
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    Blob<Dtype> *param = this->net_->learnable_params()[param_id];

    // copied from blob::Zerout
    const vector<string>& net_params_local_regular_types = this->net_->params_regularization_type();
    string regularization_type = this->param_.regularization_type();
    string local_regularization_type = net_params_local_regular_types[param_id];
    if(!local_regularization_type.empty()){
      regularization_type = local_regularization_type;
    }

    if ((regularization_type == "L1_DNS" || regularization_type == "L1_DNS_Winograd") && param->num_axes() == 4) {
      if (unthresholded_[param_id] == NULL) {
        unthresholded_[param_id] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(param->shape()));
        memcpy(unthresholded_[param_id]->mutable_cpu_data(), param->cpu_data(), param->count()*sizeof(Dtype));
        LOG(INFO) << "DNS initialized";
      }

      switch (Caffe::mode()) {
      case Caffe::CPU:
      {
        caffe_axpy<Dtype>(
          unthresholded_[param_id]->count(),
          Dtype(-1),
          static_cast<const Dtype *>(param->cpu_diff()),
          static_cast<Dtype *>(unthresholded_[param_id]->mutable_cpu_data()));
        break;
      }
      case Caffe::GPU:
      {
#ifndef CPU_ONLY
        caffe_gpu_axpy<Dtype>(
          unthresholded_[param_id]->count(),
          Dtype(-1),
          static_cast<const Dtype *>(param->gpu_diff()),
          static_cast<Dtype *>(unthresholded_[param_id]->mutable_gpu_data()));
#else
        NO_GPU;
#endif
        break;
      }
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
    }
    else {
      param->Update();
    }

    if ((regularization_type == "L1_Winograd" || regularization_type == "L1_DNS_Winograd" ||
          regularization_type == "L2_Winograd" || regularization_type == "L2_DNS_Winograd") &&
        param->num_axes() == 4 && (param->shape()[2] == 3 || param->shape()[2] == 5)) {
      if (this->param_.prune_threshold() != 0) {
        int N = param->shape()[0];
        int C = param->shape()[1];
        int K = param->shape()[2];

        WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
        int M = A->M;

        const Dtype threshold_weight = this->param_.prune_threshold();

        int mkl_max_threads_saved = mkl_get_max_threads();
        mkl_set_num_threads(1);

        switch (Caffe::mode()) {
        case Caffe::CPU:
        {
          Dtype *temp = temp_winograd_[param_id]->mutable_cpu_data();
          const Dtype *thresholds = this->param_.winograd_adjust_threshold() == 1 ? A->getNormOfInvCols()->cpu_data() : NULL;
          Dtype *wt = regularization_type == "L1_Winograd" ? param->mutable_cpu_data() : unthresholded_[param_id]->mutable_cpu_data();

          // thre(W*(G \kron G)^T)*(GGT^-1)^T
          caffe_cpu_gemm(
            CblasNoTrans, CblasTrans,
            N*C, M*M, K*K,
            (Dtype)1, wt,
            A->get()->cpu_data(),
            (Dtype)0, temp);

          for (int i = 0; i < N*C; ++i) {
            int cnt = 0;
            for (int j = 0; j < M*M; ++j) {
              Dtype thre = (thresholds ? thresholds[j] : 1)*threshold_weight;
              if (temp[i*M*M + j] <= thre && temp[i*M*M + j] >= -thre) {
                temp[i*M*M + j] = 0;
                ++cnt;
              }
            }
          }

          caffe_cpu_gemm(
            CblasNoTrans, CblasTrans,
            N*C, K*K, M*M,
            (Dtype)1, temp,
            A->getInv()->cpu_data(),
            (Dtype)0, param->mutable_cpu_data());

          break;
        }
        case Caffe::GPU:
        {
#ifndef CPU_ONLY
          Dtype *temp = temp_winograd_[param_id]->mutable_gpu_data();
          const Dtype *thresholds = (1 == this->param_.winograd_adjust_threshold()) ? A->getNormOfInvCols()->gpu_data() : NULL;
          Dtype *wt = (regularization_type == "L1_Winograd") ? param->mutable_gpu_data() : unthresholded_[param_id]->mutable_gpu_data();

          // thre(W*(G \kron G)^T)*(GGT^-1)^T
          caffe_gpu_gemm(
            CblasNoTrans, CblasTrans,
            N*C, M*M, K*K,
            (Dtype)1, wt,
            A->get()->gpu_data(),
            (Dtype)0, temp);

          if (NULL == thresholds) {
            caffe_gpu_zerout(N*C*M*M, temp, temp, threshold_weight);
          }
          else {
            caffe_gpu_zerout(N*C*M*M, temp, thresholds, M*M, threshold_weight);
          }

          caffe_gpu_gemm(
            CblasNoTrans, CblasTrans,
            N*C, K*K, M*M,
            (Dtype)1, temp,
            A->getInv()->gpu_data(),
            (Dtype)0, param->mutable_gpu_data());
#else
          NO_GPU;
#endif

          break;
        }
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }

        mkl_set_num_threads(mkl_max_threads_saved);
      } // this->param_.prune_threshold() != 0
    }
    else if (regularization_type == "ISTA") {
      Dtype weight_decay = this->param_.weight_decay();
      const vector<float>& net_params_weight_decay =
          this->net_->params_weight_decay();
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      Dtype thre = rate*local_decay;
      Dtype* data_ptr_tmp = 0;
      switch (Caffe::mode()) {
      case Caffe::CPU:
        data_ptr_tmp = static_cast<Dtype*>(param->mutable_cpu_data());
        for(int i=0;i<param->count();i++){
            if(data_ptr_tmp[i]<=thre && data_ptr_tmp[i]>=(-thre)){
                data_ptr_tmp[i]=0;
            }
            else if (data_ptr_tmp[i] > 0) {
                data_ptr_tmp[i] -= thre;
            }
            else {
                data_ptr_tmp[i] += thre;
            }
        }
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        caffe_gpu_shrinkage(param->mutable_gpu_data(),param->count(),thre);
#else
        NO_GPU;
#endif
        break;
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
    }
    else if ((regularization_type == "L1_DNS" || regularization_type == "L2_DNS") && param->num_axes() == 4) {
      // hard threshold
      Dtype thre = Dtype(this->param_.prune_threshold());
      const Dtype* data_ptr_tmp = NULL;
      Dtype* thresholded = NULL;
      switch (Caffe::mode()) {
      case Caffe::CPU:
        data_ptr_tmp = static_cast<const Dtype*>(unthresholded_[param_id]->cpu_data());
        thresholded = static_cast<Dtype *>(param->mutable_cpu_data());
        for(int i=0;i<param->count();i++){
            if(data_ptr_tmp[i]<=thre && data_ptr_tmp[i]>=(-thre)){
                thresholded[i]=0;
            }
            else {
                thresholded[i] = data_ptr_tmp[i];
            }
        }
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        caffe_gpu_zerout(param->count(), unthresholded_[param_id]->gpu_data(), param->mutable_gpu_data(), thre);
#else
        NO_GPU;
#endif
        break;
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
    }
    else if (regularization_type == "L1" || regularization_type == "L1_DNS") {
      if (regularization_type == "L1_DNS") {
        LOG(WARNING) << "L1_DNS but not for convolution";
      }
      // hard threshold
      Dtype thre_0 = Dtype(this->param_.prune_threshold());
      Dtype* data_ptr_tmp = NULL;

      Dtype *temp = NULL;
      Dtype avg_abs_diff, thre_max, alpha, beta;
      if (this->param_.max_threshold_factor() >= 2) {

        // adjust threshold based on derivatives
        // use equation th(|dL/dW_i|) = 1/(alpha*|dL/dW_i| + beta), where dL/dW_i is the partial
        // derivative of loss function w.r.t. ith parameter.
        // Use alpha and beta such that th(0) = th_max and th(|dL/dW_i|)*|dL/dW_i| < th_0
        //  -> beta = 1/th_max
        //     alpha = (1/th_0 - 1/th_max)
        thre_max = this->param_.max_threshold_factor()*thre_0;
        alpha = 1/thre_0;
        beta = 1/thre_max;
      }

      bool isWinograd =
          param->num_axes() == 4 && param->shape()[0] == param->shape()[1] &&
          (param->shape()[0] == 6 || param->shape()[0] == 8);
      const Dtype *A_kron_A_row_norm_inv = NULL;

      switch (Caffe::mode()) {
      case Caffe::CPU:
        if (isWinograd && this->param_.winograd_adjust_threshold() == 1) {
          int N = param->shape()[2];
          int C = param->shape()[3];
          int M = param->shape()[0];
          int K = M - 4 + 1;

          A_kron_A_row_norm_inv = WinogradAKronA<Dtype>::getInstance(K)->getNormRowsInv()->cpu_data();
          CHECK(param->count() == M*M*N*C);

          if (this->param_.max_threshold_factor() >= 2) {
            temp = temp_[param_id]->mutable_cpu_data();
            caffe_abs(param->count(), param->cpu_diff(), temp);
            caffe_scal(param->count(), alpha, temp);
            caffe_add_scalar(param->count(), beta, temp);
            caffe_inv(param->count(), temp, temp);
            for (int i = 0; i < M*M; ++i) {
              for (int j = 0; j < N*C; ++j) {
                temp[i*N*C + j] *= A_kron_A_row_norm_inv[i];
              }
            }
          }
        }
        else if (this->param_.max_threshold_factor() >= 2) {
          temp = temp_[param_id]->mutable_cpu_data();
          caffe_abs(param->count(), param->cpu_diff(), temp);
          caffe_scal(param->count(), alpha, temp);
          caffe_add_scalar(param->count(), beta, temp);
          caffe_inv(param->count(), temp, temp);
        }

        data_ptr_tmp = static_cast<Dtype*>(param->mutable_cpu_data());
        if (this->param_.max_threshold_factor() >= 2) {
          for(int i=0;i<param->count();i++){
            Dtype thre = temp[i];
            if(data_ptr_tmp[i]<=thre && data_ptr_tmp[i]>=(-thre)){
              data_ptr_tmp[i]=0;
            }
          }
        }
        else if (isWinograd && this->param_.winograd_adjust_threshold() == 1) {
          int N = param->shape()[2];
          int C = param->shape()[3];
          int M = param->shape()[0];
          int K = M - 4 + 1;

          for (int i = 0; i < M*M; ++i) {
            Dtype thre = A_kron_A_row_norm_inv[i]*thre_0;
            for (int j = 0; j < N*C; ++j) {
              if(data_ptr_tmp[i*N*C + j]<=thre && data_ptr_tmp[i*N*C + j]>=(-thre)){
                data_ptr_tmp[i*N*C + j] = 0;
              }
            }
          }
        }
        else {
          for(int i=0;i<param->count();i++){
            Dtype thre = thre_0;
            if(data_ptr_tmp[i]<=thre && data_ptr_tmp[i]>=(-thre)){
              data_ptr_tmp[i]=0;
            }
          }
        }
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        if (isWinograd && this->param_.winograd_adjust_threshold() == 1) {
          int N = param->shape()[2];
          int C = param->shape()[3];
          int M = param->shape()[0];
          int K = M - 4 + 1;

          A_kron_A_row_norm_inv = WinogradAKronA<Dtype>::getInstance(K)->getNormRowsInv()->cpu_data();
          CHECK(param->count() == M*M*N*C);

          if (this->param_.max_threshold_factor() >= 2) {
            temp = temp_[param_id]->mutable_gpu_data();
            caffe_gpu_abs(param->count(), param->gpu_diff(), temp);
            caffe_gpu_scal(param->count(), alpha, temp);
            caffe_gpu_add_scalar(param->count(), beta, temp);
            caffe_gpu_inv(param->count(), temp, temp);
            for (int i = 0; i < M*M; ++i) {
              caffe_gpu_scal(N*C, A_kron_A_row_norm_inv[i], temp + i*N*C);
            }

            caffe_gpu_zerout(param->count(), param->mutable_gpu_data(), temp, param->count(), (Dtype)1);
          }
          else {
            for (int i = 0; i < M*M; ++i) {
              caffe_gpu_zerout(param->mutable_gpu_data() + i*N*C, N*C, A_kron_A_row_norm_inv[i]*thre_0);
            }
          }
        }
        else if (this->param_.max_threshold_factor() >= 2) {
          temp = temp_[param_id]->mutable_gpu_data();
          caffe_gpu_abs(param->count(), param->gpu_diff(), temp);
          caffe_gpu_scal(param->count(), alpha, temp);
          caffe_gpu_add_scalar(param->count(), beta, temp);
          caffe_gpu_inv(param->count(), temp, temp);

          caffe_gpu_zerout(param->count(), param->mutable_gpu_data(), temp, param->count(), (Dtype)1);
        }
        else {
          caffe_gpu_zerout(param->mutable_gpu_data(),param->count(),thre_0);
        }
#else
        NO_GPU;
#endif
        break;
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
      //this->net_->learnable_params()[param_id]->Zerout();
    }
    else if (regularization_type == "L2" || regularization_type == "L2_DNS") {
      if (regularization_type == "L2_DNS") {
        LOG(WARNING) << "L2_DNS but not for convolution";
      }
    }
    else {
      LOG(FATAL) << "Unknown regularization type: " << regularization_type;
    }
  } /* for each parameter */
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  const vector<string>& net_params_local_regular_types = this->net_->params_regularization_type();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  string local_regularization_type = net_params_local_regular_types[param_id];
  if(!local_regularization_type.empty()){
	  regularization_type = local_regularization_type;
  }
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  Dtype regularization_term = Dtype(0);

  Blob<Dtype> *param = net_params[param_id];
  if ((regularization_type == "L1_Winograd" || regularization_type == "L1_DNS_Winograd") &&
      param->num_axes() == 4 && (param->shape()[2] == 3 || param->shape()[2] == 5)) {

    assert(param->shape()[2] == param->shape()[3]);
    assert(temp_[param_id]->num_axes() == 4);
    assert(temp_[param_id]->shape()[0] == param->shape()[0]);
    assert(temp_[param_id]->shape()[1] == param->shape()[1]);
    assert(temp_[param_id]->shape()[2] == param->shape()[2]);
    assert(temp_[param_id]->shape()[3] == param->shape()[3]);

    int N = param->shape()[0];
    int C = param->shape()[1];
    int K = param->shape()[2];

    WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
    int M = A->M;
    const Dtype threshold_weight = this->param_.prune_threshold();

    switch (Caffe::mode()) {
      case Caffe::CPU: {
        Dtype *temp = temp_winograd_[param_id]->mutable_cpu_data();

        caffe_cpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, M*M, K*K,
          (Dtype)1, param->cpu_data(),
          A->get()->cpu_data(),
          (Dtype)0, temp);

        regularization_term = caffe_cpu_asum(N*C*M*M, temp);

        caffe_cpu_sign(N*C*M*M, temp, temp);
        caffe_cpu_gemm(
          CblasNoTrans, CblasNoTrans,
          N*C, K*K, M*M,
          (Dtype)1, temp,
          A->get()->cpu_data(),
          (Dtype)0, temp_[param_id]->mutable_cpu_data());

        // sign(W*(G \kron G)^T)*(G \kron G)
        //
        caffe_axpy(param->count(),
            (Dtype)local_decay,
            temp_[param_id]->cpu_data(),
            param->mutable_cpu_diff());

        break;
      }
      case Caffe::GPU: {
#ifndef CPU_ONLY
        Dtype *temp = temp_winograd_[param_id]->mutable_gpu_data();

        caffe_gpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, M*M, K*K,
          (Dtype)1, param->gpu_data(),
          A->get()->gpu_data(),
          (Dtype)0, temp);

        caffe_gpu_asum(N*C*M*M, temp, &regularization_term);

        caffe_gpu_sign(N*C*M*M, temp, temp);
        caffe_gpu_gemm(
          CblasNoTrans, CblasNoTrans,
          N*C, K*K, M*M,
          (Dtype)1, temp,
          A->get()->gpu_data(),
          (Dtype)0, temp_[param_id]->mutable_gpu_data());

        // sign(W*(G \kron G)^T)*(G \kron G)
        //
        caffe_gpu_axpy(param->count(),
            (Dtype)local_decay,
            temp_[param_id]->gpu_data(),
            param->mutable_gpu_diff());
#else
        NO_GPU;
#endif

        break;
      }
      default :
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }

    regularization_term *= local_decay*K*K/M/M;
  }
  else if ((regularization_type == "L2_Winograd" || regularization_type == "L2_DNS_Winograd") &&
      param->num_axes() == 4 && (param->shape()[2] == 3 || param->shape()[2] == 5)) {
    NOT_IMPLEMENTED;
  }
  else {
    switch (Caffe::mode()) {
    case Caffe::CPU: {
      if (local_decay) {
        if (regularization_type == "L2" || regularization_type == "L2_DNS") {

          // add weight decay
          caffe_axpy(param->count(),
              local_decay,
              param->cpu_data(),
              param->mutable_cpu_diff());
          //calcuate the l2 regularization term
          regularization_term = caffe_cpu_dot(
              param->count(),
              param->cpu_data(),
              param->cpu_data());

          if (isnan(regularization_term)) {
            ostringstream str;
            str << "data:sign:diff";
            for (int i = 0; i < param->count(); ++i) {
              str << " " << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
            }
            LOG(INFO) << str.str();

            str.str("");
            str << "nan";
            for (int i = 0; i < param->count(); ++i) {
              if(isnan(param->cpu_data()[i])) {
                str << " " << i << ":" << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
              }
            }
            LOG(INFO) << str.str();

            LOG(FATAL) << "nan at param " << param_id;
          }
          regularization_term *= local_decay/(Dtype)2.0;
        } else if (regularization_type == "L1" || regularization_type == "ISTA" || regularization_type == "L1_DNS" || regularization_type == "L1_Winograd") {
          caffe_cpu_sign(param->count(),
              param->cpu_data(),
              temp_[param_id]->mutable_cpu_data());
          caffe_axpy(param->count(),
              local_decay,
              temp_[param_id]->cpu_data(),
              param->mutable_cpu_diff()); // JSP: regularization term applied here
          //calcuate the l1 regularization term
          regularization_term = caffe_cpu_asum(param->count(), param->cpu_data());
          if (isnan(regularization_term)) {
            ostringstream str;
            str << "data:sign:diff";
            for (int i = 0; i < param->count(); ++i) {
              str << " " << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
            }
            LOG(INFO) << str.str();

            str.str("");
            str << "nan";
            for (int i = 0; i < param->count(); ++i) {
              if(isnan(param->cpu_data()[i])) {
                str << " " << i << ":" << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
              }
            }
            LOG(INFO) << str.str();

            LOG(FATAL) << "nan at param " << param_id;
          }
          regularization_term *= local_decay;
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      if (local_decay) {
        if (regularization_type == "L2" || regularization_type == "L2_DNS") {
          // add weight decay
          caffe_gpu_axpy(param->count(),
              local_decay,
              param->gpu_data(),
              param->mutable_gpu_diff());
          //calcuate the l2 regularization term
          caffe_gpu_dot(param->count(), param->gpu_data(), param->gpu_data(), &regularization_term);
          if (isnan(regularization_term)) {
            ostringstream str;
            str << "data:sign:diff";
            for (int i = 0; i < param->count(); ++i) {
              str << " " << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
            }
            LOG(INFO) << str.str();

            str.str("");
            str << "nan";
            for (int i = 0; i < param->count(); ++i) {
              if(isnan(param->cpu_data()[i])) {
                str << " " << i << ":" << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
              }
            }
            LOG(INFO) << str.str();

            LOG(FATAL) << "nan at param " << param_id;
          }
          regularization_term *= local_decay/(Dtype)2.0;
        } else if (regularization_type == "L1" || regularization_type == "ISTA" || regularization_type == "L1_DNS" || regularization_type == "L1_Winograd") {
          caffe_gpu_sign(param->count(),
              param->gpu_data(),
              temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(param->count(),
              local_decay,
              temp_[param_id]->gpu_data(),
              param->mutable_gpu_diff());
          //calcuate the l1 regularization term
          caffe_gpu_asum(param->count(), param->gpu_data(), &regularization_term);
          if (isnan(regularization_term)) {
            ostringstream str;
            str << "data:sign:diff";
            for (int i = 0; i < param->count(); ++i) {
              str << " " << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
            }
            LOG(INFO) << str.str();

            str.str("");
            str << "nan";
            for (int i = 0; i < param->count(); ++i) {
              if(isnan(param->cpu_data()[i])) {
                str << " " << i << ":" << param->cpu_data()[i] << ":" << temp_[param_id]->cpu_data()[i] << ":" << param->cpu_diff()[i];
              }
            }
            LOG(INFO) << str.str();

            LOG(FATAL) << "nan at param " << param_id;
          }
          regularization_term *= local_decay;
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
  return regularization_term;
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetSparsity(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sparsity = Dtype(0);
  Blob<Dtype> *param = net_params[param_id];

  if (param->num_axes() == 4 && param->shape(0) == param->shape(1) && (param->shape(0) == 6 || param->shape(0) == 8)) {
    // Winograd layer
    int N = param->shape()[2];
    int C = param->shape()[3];
    int K = param->shape()[0] - 4 + 1;

    WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
    int M = A->M;

    switch (Caffe::mode()) {
    case Caffe::CPU: {
      caffe_cpu_gemm(
        CblasTrans, CblasTrans,
        N*C, K*K, M*M,
        (Dtype)1, param->cpu_data(),
        A->getInv()->cpu_data(),
        (Dtype)0, temp_[param_id]->mutable_cpu_data());
      caffe_cpu_if_zerout(N*C*K*K,
        temp_[param_id]->cpu_data(),
        temp_[param_id]->mutable_cpu_data(),
        (Dtype)this->param_.measure_threshold());
      sparsity = caffe_cpu_asum(N*C*K*K, temp_[param_id]->cpu_data())*Dtype(100)/(N*C*K*K);
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      caffe_gpu_gemm(
        CblasTrans, CblasTrans,
        N*C, K*K, M*M,
        (Dtype)1, param->gpu_data(),
        A->getInv()->gpu_data(),
        (Dtype)0, temp_[param_id]->mutable_gpu_data());
      caffe_gpu_if_zerout(N*C*K*K,
        temp_[param_id]->gpu_data(),
        temp_[param_id]->mutable_gpu_data(),
        (Dtype)this->param_.measure_threshold());
      caffe_gpu_asum(N*C*K*K,temp_[param_id]->gpu_data(),&sparsity);
      sparsity = sparsity*Dtype(100)/(N*C*K*K);
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
  else {
    switch (Caffe::mode()) {
    case Caffe::CPU: {
      caffe_cpu_if_zerout(param->count(),
        param->cpu_data(),
        temp_[param_id]->mutable_cpu_data(),
        (Dtype)this->param_.measure_threshold());
      //calcuate the sparsity
      sparsity = caffe_cpu_asum(param->count(),temp_[param_id]->cpu_data())*Dtype(100)/param->count();
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      caffe_gpu_if_zerout(param->count(),
        param->gpu_data(),
        temp_[param_id]->mutable_gpu_data(),
        (Dtype)this->param_.measure_threshold());
      caffe_gpu_asum(param->count(),temp_[param_id]->gpu_data(),&sparsity);
      sparsity = sparsity*Dtype(100)/param->count();
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }

  return sparsity;
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetWinogradSparsityOld(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sparsity = Dtype(0);
  Blob<Dtype> *param = net_params[param_id];

  const vector<string>& net_params_local_regular_types = this->net_->params_regularization_type();
  string regularization_type = this->param_.regularization_type();
  string local_regularization_type = net_params_local_regular_types[param_id];
  if(!local_regularization_type.empty()){
    regularization_type = local_regularization_type;
  }

  if (param->num_axes() == 4 && param->shape(2) == param->shape(3) &&
      (param->shape(2) == 3 || param->shape(2) == 5) &&
      (regularization_type == "L1_Winograd" || regularization_type == "L2_Winograd")) {
    int N = param->shape()[0];
    int C = param->shape()[1];
    int K = param->shape()[2];

    WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
    int M = A->M;

    switch (Caffe::mode()) {
    case Caffe::CPU: {
      Dtype *winograd_weights = temp_winograd_[param_id]->mutable_cpu_data();
      const Dtype *thresholds = this->param_.winograd_adjust_threshold() == 1 ? A->getNormOfInvCols()->cpu_data() : NULL;

      caffe_cpu_gemm(
        CblasNoTrans, CblasTrans,
        N*C, M*M, K*K,
        (Dtype)1, param->cpu_data(),
        A->get()->cpu_data(),
        (Dtype)0, winograd_weights);

      int count = 0;
      for (int i = 0; i < N*C; ++i) {
        for (int j = 0; j < M*M; ++j) {
          Dtype thre = (thresholds ? thresholds[j] : 1)*this->param_.measure_threshold();
          if (winograd_weights[i*M*M + j] <= thre && winograd_weights[i*M*M + j] >= -thre) {
            ++count;
          }
        }
      }
      sparsity = count*100./(N*C*M*M);
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      Dtype *winograd_weights = temp_winograd_[param_id]->mutable_gpu_data();
      const Dtype *thresholds = this->param_.winograd_adjust_threshold() == 1 ? A->getNormOfInvCols()->gpu_data() : NULL;

      caffe_gpu_gemm(
        CblasNoTrans, CblasTrans,
        N*C, M*M, K*K,
        (Dtype)1, param->gpu_data(),
        A->get()->gpu_data(),
        (Dtype)0, winograd_weights);

      if (NULL == thresholds) {
        caffe_gpu_if_zerout(N*C*M*M, winograd_weights, winograd_weights, (Dtype)this->param_.measure_threshold());
      }
      else {
        caffe_gpu_if_zerout(N*C*M*M, winograd_weights, winograd_weights, thresholds, M*M, (Dtype)this->param_.measure_threshold());
      }
      caffe_gpu_asum(N*C*M*M, winograd_weights, &sparsity);
      sparsity = sparsity*100./(N*C*M*M);
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
  else {
    return GetWinogradSparsity(param_id);
  }

  return sparsity;
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetWinogradSparsity(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sparsity = Dtype(0);
  Blob<Dtype> *param = net_params[param_id];

  const vector<string>& net_params_local_regular_types = this->net_->params_regularization_type();
  string regularization_type = this->param_.regularization_type();
  string local_regularization_type = net_params_local_regular_types[param_id];
  if(!local_regularization_type.empty()){
    regularization_type = local_regularization_type;
  }

  if (param->num_axes() == 4 && param->shape()[2] == param->shape()[3] &&
      (param->shape()[2] == 3 || param->shape()[2] == 5) &&
      (regularization_type == "L1_Winograd" || regularization_type == "L2_Winograd")) {
    int N = param->shape()[0];
    int C = param->shape()[1];
    int K = param->shape()[2];

    WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
    int M = A->M;
    Dtype thre = this->param_.measure_threshold();

    switch (Caffe::mode()) {
    case Caffe::CPU: {
      const Dtype *weights = param->cpu_data();
      Dtype *temp = temp_[param_id]->mutable_cpu_data();

      if (this->param_.prune_threshold() == 0) {
        Dtype *temp_winograd = temp_winograd_[param_id]->mutable_cpu_data();
        const Dtype *thresholds = this->param_.winograd_adjust_threshold() == 1 ? A->getNormOfInvCols()->cpu_data() : NULL;
        Dtype *wt = /*regularization_type == "L1_Winograd" ? */param->mutable_cpu_data();// : unthresholded_[param_id]->mutable_cpu_data();
        memcpy(temp, wt, sizeof(Dtype)*(N*C)*(K*K));

        // thre(W*(G \kron G)^T)*(GGT^-1)^T
        caffe_cpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, M*M, K*K,
          (Dtype)1, temp,
          A->get()->cpu_data(),
          (Dtype)0, temp_winograd);

        for (int i = 0; i < N*C; ++i) {
          int cnt = 0;
          for (int j = 0; j < M*M; ++j) {
            Dtype thre = (thresholds ? thresholds[j] : 1)*thre;
            if (temp_winograd[i*M*M + j] <= thre && temp_winograd[i*M*M + j] >= -thre) {
              temp_winograd[i*M*M + j] = 0;
              ++cnt;
            }
          }
        }

        caffe_cpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, K*K, M*M,
          (Dtype)1, temp_winograd,
          A->getInv()->cpu_data(),
          (Dtype)0, temp);
      }

      Dtype *winograd_weights = temp_winograd_[param_id]->mutable_cpu_data();
      caffe_cpu_gemm(
        CblasNoTrans, CblasTrans,
        N*C, M*M, K*K,
        (Dtype)1, this->param_.prune_threshold() == 0 ? temp : weights,
        A->get()->cpu_data(),
        (Dtype)0, winograd_weights);

      int count = 0;
      for (int i = 0; i < (N*C)*(M*M); ++i) {
        if (winograd_weights[i] == 0) ++count;
      }

      sparsity = count*100./(N*C*M*M);
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      if (this->param_.prune_threshold() == 0) {
        Dtype *temp_winograd = temp_winograd_[param_id]->mutable_gpu_data();
        const Dtype *thresholds = (1 == this->param_.winograd_adjust_threshold()) ? A->getNormOfInvCols()->gpu_data() : NULL;
        Dtype *wt = /*(regularization_type == "L1_Winograd") ? */param->mutable_gpu_data();// : unthresholded_[param_id]->mutable_gpu_data();
        CUDA_CHECK(cudaMemcpy(temp_[param_id]->mutable_gpu_data(), wt, sizeof(Dtype)*(N*C)*(K*K), cudaMemcpyDeviceToDevice));

        // thre(W*(G \kron G)^T)*(GGT^-1)^T
        caffe_gpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, M*M, K*K,
          (Dtype)1, temp_[param_id]->gpu_data(),
          A->get()->gpu_data(),
          (Dtype)0, temp_winograd);

        if (NULL == thresholds) {
          caffe_gpu_zerout(N*C*M*M, temp_winograd, temp_winograd, thre);
        }
        else {
          caffe_gpu_zerout(N*C*M*M, temp_winograd, thresholds, M*M, thre);
        }

        caffe_gpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, K*K, M*M,
          (Dtype)1, temp_winograd,
          A->getInv()->gpu_data(),
          (Dtype)0, temp_[param_id]->mutable_gpu_data());
      }

      Dtype *winograd_weights = temp_winograd_[param_id]->mutable_gpu_data();
      caffe_gpu_gemm(
        CblasNoTrans, CblasTrans,
        N*C, M*M, K*K,
        (Dtype)1, this->param_.prune_threshold() == 0 ? temp_[param_id]->mutable_gpu_data() : param->gpu_data(),
        A->get()->gpu_data(),
        (Dtype)0, winograd_weights);

      caffe_gpu_if_zerout(N*C*M*M, winograd_weights, winograd_weights, (Dtype)0);
      caffe_gpu_asum(N*C*M*M, winograd_weights, &sparsity);
      sparsity = sparsity*100./(N*C*M*M);
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
  else if (param->num_axes() == 4 && param->shape()[0] == param->shape()[1] && (param->shape()[0] == 6 || param->shape()[0] == 8)) {
    // Winograd layer
    switch (Caffe::mode()) {
    case Caffe::CPU: {
      caffe_cpu_if_zerout(param->count(),
        param->cpu_data(),
        temp_[param_id]->mutable_cpu_data(),
        (Dtype)this->param_.measure_threshold());
      //calcuate the sparsity
      sparsity = caffe_cpu_asum(param->count(),temp_[param_id]->cpu_data())*Dtype(100)/param->count();
      break;
    }
    case Caffe::GPU: {
#ifndef CPU_ONLY
      caffe_gpu_if_zerout(param->count(),
        param->gpu_data(),
        temp_[param_id]->mutable_gpu_data(),
        (Dtype)this->param_.measure_threshold());
      caffe_gpu_asum(param->count(),temp_[param_id]->gpu_data(),&sparsity);
      sparsity = sparsity*Dtype(100)/param->count();
#else
      NO_GPU;
#endif
      break;
    }
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
  else {
    return GetSparsity(param_id);
  }

  return sparsity;
}

template <typename Dtype>
void SGDSolver<Dtype>::PrintWinogradFiberSliceSparsity() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();

  const vector<string>& net_params_local_regular_types = this->net_->params_regularization_type();
  string regularization_type = this->param_.regularization_type();

  ostringstream sparsity_msg_stream;
  sparsity_msg_stream << "     Winograd fiber/slice sparsity %: \n";
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    Blob<Dtype> *param = this->net_->learnable_params()[param_id];

    string local_regularization_type = net_params_local_regular_types[param_id];
    if(!local_regularization_type.empty()){
      regularization_type = local_regularization_type;
    }

    if (param->num_axes() == 4 && param->shape()[2] == param->shape()[3] &&
        (param->shape()[2] == 3 || param->shape()[2] == 5) &&
        (regularization_type == "L1_Winograd" || regularization_type == "L2_Winograd")) {

      int N = param->shape()[0];
      int C = param->shape()[1];
      int K = param->shape()[2];

      WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
      int M = A->M;

      Dtype thre = this->param_.measure_threshold();

      switch (Caffe::mode()) {
      case Caffe::CPU: {
        Dtype *winograd_weights = temp_winograd_[param_id]->mutable_cpu_data();

        caffe_cpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, M*M, K*K,
          (Dtype)1, param->cpu_data(),
          A->get()->cpu_data(),
          (Dtype)0, winograd_weights);

        sparsity_msg_stream <<
          100*caffe_cpu_fiber_sparsity(N, C, M*M, winograd_weights, 0, thre) << "/" << 
          100*caffe_cpu_slice_sparsity(N, C, M*M, winograd_weights, 0, thre) << "\t" <<
          100*caffe_cpu_fiber_sparsity(N, C, M*M, winograd_weights, 1, thre) << "/" << 
          100*caffe_cpu_slice_sparsity(N, C, M*M, winograd_weights, 1, thre) << "\t" <<
          100*caffe_cpu_fiber_sparsity(N, C, M*M, winograd_weights, 2, thre) << "/" << 
          100*caffe_cpu_slice_sparsity(N, C, M*M, winograd_weights, 2, thre) << "\n";
          break;
      }
      case Caffe::GPU: {
#ifndef CPU_ONLY
        Dtype *winograd_weights = temp_winograd_[param_id]->mutable_gpu_data();

        caffe_gpu_gemm(
          CblasNoTrans, CblasTrans,
          N*C, M*M, K*K,
          (Dtype)1, param->gpu_data(),
          A->get()->gpu_data(),
          (Dtype)0, winograd_weights);

        const Dtype *winograd_weights2 = temp_winograd_[param_id]->cpu_data();
        sparsity_msg_stream <<
          100*caffe_cpu_fiber_sparsity(N, C, M*M, winograd_weights2, 0, thre) << "/" << 
          100*caffe_cpu_slice_sparsity(N, C, M*M, winograd_weights2, 0, thre) << "\t" <<
          100*caffe_cpu_fiber_sparsity(N, C, M*M, winograd_weights2, 1, thre) << "/" << 
          100*caffe_cpu_slice_sparsity(N, C, M*M, winograd_weights2, 1, thre) << "\t" <<
          100*caffe_cpu_fiber_sparsity(N, C, M*M, winograd_weights2, 2, thre) << "/" << 
          100*caffe_cpu_slice_sparsity(N, C, M*M, winograd_weights2, 2, thre) << "\n";
#else
          NO_GPU;
#endif
          break;
      }
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
    }
  }
  LOG(INFO) << sparsity_msg_stream.str();
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetGroupSparsity(int param_id, bool dimen) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  return 100*caffe_cpu_group_sparsity(net_params[param_id]->shape(0),
		  net_params[param_id]->count()/net_params[param_id]->shape(0),
		  net_params[param_id]->cpu_data(),
		  dimen);
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetFiberSparsity(int param_id, int mode) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  return 100*caffe_cpu_fiber_sparsity(
      net_params[param_id]->shape(0),
      net_params[param_id]->shape(1),
      net_params[param_id]->count()/(net_params[param_id]->shape(0)*net_params[param_id]->shape(1)),
		  net_params[param_id]->cpu_data(),
		  mode,
		  (Dtype)this->param_.measure_threshold());
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetSliceSparsity(int param_id, int mode) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  return 100*caffe_cpu_slice_sparsity(
      net_params[param_id]->shape(0),
      net_params[param_id]->shape(1),
      net_params[param_id]->count()/(net_params[param_id]->shape(0)*net_params[param_id]->shape(1)),
		  net_params[param_id]->cpu_data(),
		  mode,
		  (Dtype)this->param_.measure_threshold());
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetGroupSparsity(int param_id, int ydimen,int xdimen) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  int equivalent_ch = net_params[param_id]->count()/net_params[param_id]->shape(0);
  CHECK_EQ(net_params[param_id]->shape(0)%ydimen,0);
  CHECK_EQ(equivalent_ch%xdimen,0);
  int block_num_x = equivalent_ch/xdimen;
  int block_num_y = net_params[param_id]->shape(0)/ydimen;
  int count = 0;
  for(int by=0;by<block_num_y;by++){
	  for(int bx=0;bx<block_num_x;bx++){
		  count++;
		  bool inner_break = false;
		  for(int y=0;y<ydimen;y++){
			  if(inner_break) break;
		  	  for(int x=0;x<xdimen;x++){
		  		  int idx = (by*ydimen+y)*equivalent_ch + (bx*xdimen+x);
		  		  if(net_params[param_id]->cpu_data()[idx]){
		  			  count--;
		  			  inner_break = true;
		  			  break;
		  		  }
		      }
		  }
	  }
  }
  return (Dtype)(100*count)/(Dtype)(block_num_x*block_num_y);
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GroupLassoRegularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<int >& net_param_groups = this->net_->param_groups();
  const vector<float>& net_params_breadth_decay_multi =
  	             this->net_->params_breadth_decay();
  const vector<float>& net_params_kernel_shape_decay_multi =
  	             this->net_->params_kernel_shape_decay();
  const vector<BlockGroupLassoSpec> net_params_block_group_lasso =
    	             this->net_->params_block_group_lasso()[param_id];
  Dtype local_breadth_decay = this->param_.breadth_decay() * net_params_breadth_decay_multi[param_id];
  Dtype local_kernel_shape_decay = this->param_.kernel_shape_decay() * net_params_kernel_shape_decay_multi[param_id];
  Dtype regularization_term = Dtype(0);
  bool if_learn_kernel_shape = local_kernel_shape_decay!=0;// && (net_params[param_id]->num_axes()==4);
  bool if_learn_breadth = local_breadth_decay!=0;// && (net_params[param_id]->num_axes()==4 );
  int equivalent_ch = net_params[param_id]->count()/net_params[param_id]->shape(0);
  switch (Caffe::mode()) {
  case Caffe::CPU: {

	if(if_learn_breadth || if_learn_kernel_shape){
		LOG(FATAL)<< "Deprecated in CPU mode: breadth and kernel shape decay (use block group decay instead)";
	}

	for (int blk_idx=0;blk_idx<net_params_block_group_lasso.size();blk_idx++){
		int xdimen = net_params_block_group_lasso[blk_idx].xdimen();
		int ydimen = net_params_block_group_lasso[blk_idx].ydimen();
		Dtype block_decay_mult = net_params_block_group_lasso[blk_idx].block_decay_mult();
		Dtype local_block_group_decay = block_decay_mult*this->param_.block_group_decay();
		if(local_block_group_decay){
			caffe_cpu_block_group_lasso(
					net_params[param_id]->shape(0),
					equivalent_ch,
					ydimen, xdimen,
					net_params[param_id]->cpu_data(),
					temp_[param_id]->mutable_cpu_data());
			Dtype term;
			term = caffe_cpu_asum(temp_[param_id]->count(),temp_[param_id]->cpu_data());
			term /= (xdimen*ydimen);
			regularization_term += term*local_block_group_decay;

			caffe_div_checkzero(net_params[param_id]->count(),
				  net_params[param_id]->cpu_data(),
				  temp_[param_id]->cpu_data(),
				  temp_[param_id]->mutable_cpu_data());
		    caffe_axpy(net_params[param_id]->count(),
				  local_block_group_decay,
				  temp_[param_id]->cpu_data(),
				  net_params[param_id]->mutable_cpu_diff());
		}
	}

	/*
    if (if_learn_kernel_shape) {
      if((net_params[param_id]->shape(2)>1) || (net_params[param_id]->shape(3)>1) || net_param_groups[param_id]>1){
    	  LOG(FATAL)<< "Unsupported in CPU mode: group lasso for convolutional layers with kernel > 1x1 or with more than 1 kernel bank";
      }

      for(int c=0;c<net_params[param_id]->shape(1);c++){
    	  Dtype tmp = caffe_cpu_strided_dot(net_params[param_id]->shape(0),
    			  net_params[param_id]->cpu_data()+c,net_params[param_id]->shape(1),
    			  net_params[param_id]->cpu_data()+c,net_params[param_id]->shape(1));
		  tmp = sqrt(tmp);
		  regularization_term += tmp;
		  temp_[param_id]->mutable_cpu_data()[c] = tmp;
      }
      regularization_term *= local_breadth_decay;
      //copy memory
      for(int num=1;num<net_params[param_id]->shape(0);num++){
    	  memcpy(temp_[param_id]->mutable_cpu_data()+num*net_params[param_id]->shape(1),
    			  temp_[param_id]->cpu_data(),
    			  net_params[param_id]->shape(1)*sizeof(Dtype));
      }
      caffe_div_checkzero(net_params[param_id]->count(),
    		  net_params[param_id]->cpu_data(),
    		  temp_[param_id]->cpu_data(),
    		  temp_[param_id]->mutable_cpu_data());
      caffe_axpy(net_params[param_id]->count(),
    		  	  local_breadth_decay,
    		  	  temp_[param_id]->cpu_data(),
                  net_params[param_id]->mutable_cpu_diff());
    }
    */
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (this->param_.regularization_type() == "L1_Winograd" && net_params[param_id]->num_axes() == 4 &&
        (net_params[param_id]->shape(2) == 3 || net_params[param_id]->shape(2) == 5)) {
      if (if_learn_kernel_shape) {
        // group lasso of all-zero ic-mode slices
      }
      if (if_learn_breadth) {
        // group lasso of all-zero oc-mode slices
      }
    } // Winograd
    else {
	//group lasso along columns (channels)
    if (if_learn_kernel_shape) {
    	int group_size = net_params[param_id]->shape(0)/net_param_groups[param_id];//number of kernels in each group
    	for (int g=0;g<net_param_groups[param_id];g++){
    		int offset = g*group_size*equivalent_ch;
    		caffe_gpu_bar_group_lasso(group_size,
					equivalent_ch,
					net_params[param_id]->gpu_data()+offset,
					temp_[param_id]->mutable_gpu_data()+offset, true);//get the denominator of each w
			Dtype term;
			caffe_gpu_asum(equivalent_ch,temp_[param_id]->gpu_data()+offset,&term);
			regularization_term += term*local_kernel_shape_decay;
    	}
    	caffe_gpu_div_checkzero(net_params[param_id]->count(), net_params[param_id]->gpu_data(), temp_[param_id]->gpu_data(), temp_[param_id]->mutable_gpu_data());
		caffe_gpu_axpy(net_params[param_id]->count(),
					local_kernel_shape_decay,
					temp_[param_id]->gpu_data(),
					net_params[param_id]->mutable_gpu_diff());
    }

    //group lasso along rows (kernels)
    if (if_learn_breadth) {
		int group_size = net_params[param_id]->shape(0)/net_param_groups[param_id];//number of kernels in each group
		for (int g=0;g<net_param_groups[param_id];g++){
			int offset = g*group_size*equivalent_ch;
			caffe_gpu_bar_group_lasso(group_size,
					equivalent_ch,
					net_params[param_id]->gpu_data()+offset,
					temp_[param_id]->mutable_gpu_data()+offset, false);//get the denominator of each w
			Dtype term;
			caffe_gpu_asum(group_size,temp_[param_id]->gpu_data()+offset,&term,equivalent_ch);
			regularization_term += term*local_breadth_decay;
		}
		caffe_gpu_div_checkzero(net_params[param_id]->count(), net_params[param_id]->gpu_data(), temp_[param_id]->gpu_data(), temp_[param_id]->mutable_gpu_data());
		caffe_gpu_axpy(net_params[param_id]->count(),
					local_breadth_decay,
					temp_[param_id]->gpu_data(),
					net_params[param_id]->mutable_gpu_diff());
	}

    for (int blk_idx=0;blk_idx<net_params_block_group_lasso.size();blk_idx++){
    	int xdimen = net_params_block_group_lasso[blk_idx].xdimen();
    	int ydimen = net_params_block_group_lasso[blk_idx].ydimen();
    	Dtype block_decay_mult = net_params_block_group_lasso[blk_idx].block_decay_mult();
    	Dtype local_block_group_decay = block_decay_mult*this->param_.block_group_decay();
    	if(local_block_group_decay){
			caffe_gpu_block_group_lasso(
					net_params[param_id]->shape(0),
					equivalent_ch,
					ydimen, xdimen,
					net_params[param_id]->gpu_data(),
					temp_[param_id]->mutable_gpu_data());
			Dtype term;
			caffe_gpu_asum(temp_[param_id]->count(),temp_[param_id]->gpu_data(),&term);
			term /= (xdimen*ydimen);
			regularization_term += term*local_block_group_decay;

			caffe_gpu_div_checkzero(net_params[param_id]->count(), net_params[param_id]->gpu_data(), temp_[param_id]->gpu_data(), temp_[param_id]->mutable_gpu_data());
			caffe_gpu_axpy(net_params[param_id]->count(),
						local_block_group_decay,
						temp_[param_id]->gpu_data(),
						net_params[param_id]->mutable_gpu_diff());
    	}
    }
    } // !Winograd
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }

  return regularization_term;
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
