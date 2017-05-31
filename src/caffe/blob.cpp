#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/mmio.hpp"
#include "caffe/util/winograd.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  bool actual_reshaping = false;
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    if (shape_[i] != shape[i]) {
      actual_reshaping = true;
      shape_[i] = shape[i];
    }
    shape_data[i] = shape[i];
  }
  // We restart sync objects when there was change of shape
  // requested count is bgger than current capacity
  if ( (actual_reshaping == true) || (count_ > capacity_) ) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    if (connectivity_.get() != NULL) {
      connectivity_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    }
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_diff(Dtype* diff) {
  CHECK(diff);
  diff_->set_cpu_data(diff);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_connectivity() const {
  CHECK(connectivity_);
  return (const Dtype*)connectivity_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_connectivity() const {
  CHECK(connectivity_);
  return (const Dtype*)connectivity_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

// begin Intel caffe

template <typename Dtype>
const Dtype* Blob<Dtype>::prv_data() const {
  CHECK(data_);
  return (const Dtype*)data_->prv_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_prv_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_prv_data());
}

template <typename Dtype>
const Dtype* Blob<Dtype>::prv_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->prv_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_prv_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_prv_data());
}


template <typename Dtype>
void Blob<Dtype>::set_prv_data_descriptor(shared_ptr<PrvMemDescr> descriptor,
         bool same_data) {
    CHECK(data_);
    data_->set_prv_descriptor(descriptor, same_data);
}

template <typename Dtype>
void Blob<Dtype>::set_prv_diff_descriptor(shared_ptr<PrvMemDescr> descriptor,
                 bool same_data) {
  CHECK(diff_);
  diff_->set_prv_descriptor(descriptor, same_data);
}

template <typename Dtype>
shared_ptr<PrvMemDescr> Blob<Dtype>::get_prv_data_descriptor() {
  CHECK(data_);
  return data_->prv_descriptor_;
}

template <typename Dtype>
shared_ptr<PrvMemDescr> Blob<Dtype>::get_prv_diff_descriptor() {
  CHECK(diff_);
  return diff_->prv_descriptor_;
}

// end Intel caffe

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_connectivity() {
  CHECK(connectivity_);
  return static_cast<Dtype*>(connectivity_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_connectivity() {
  CHECK(connectivity_);
  return static_cast<Dtype*>(connectivity_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
  connectivity_ = other.connectivity();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<long>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
    if ((diff_->head() == SyncedMemory::SYNCED_PRV) ||
        (diff_->head() == SyncedMemory::HEAD_AT_PRV)) {
      CHECK_EQ(true, get_prv_data_descriptor()->layout_compare(
                get_prv_diff_descriptor()));
      caffe_axpy<Dtype>(prv_diff_count(), Dtype(-1),
          static_cast<const Dtype*>(diff_->prv_data()),
          static_cast<Dtype*>(data_->mutable_prv_data()));
      break;
    }
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    if (connectivity_.get()) {
      caffe_cpu_eltwise_multi(count_,
          static_cast<const Dtype*>(connectivity_->cpu_data()),
          static_cast<Dtype*>(diff_->mutable_cpu_data()) );
    }
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    if (connectivity_.get()) {
      caffe_gpu_eltwise_multi(count_,
          static_cast<const Dtype*>(connectivity_->gpu_data()),
          static_cast<Dtype*>(diff_->mutable_gpu_data()) );
    }
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}


template <typename Dtype>
void Blob<Dtype>::Zerout(Dtype thre) {
  // Zero out elements whose values are smaller than thre.
  Dtype* data_ptr_tmp = 0;
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    //caffe_axpy<Dtype>(count_, Dtype(-1),
    //    static_cast<const Dtype*>(diff_->cpu_data()),
    //    static_cast<Dtype*>(data_->mutable_cpu_data()));
	  data_ptr_tmp = static_cast<Dtype*>(data_->mutable_cpu_data());
	  for(int i=0;i<count_;i++){
		  if(data_ptr_tmp[i]<=thre && data_ptr_tmp[i]>=(-thre)){
			  data_ptr_tmp[i]=0;
		  }
	  }
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform zerout on GPU
	  //data_ptr_tmp = static_cast<Dtype*>(data_->mutable_gpu_data());
	  //	  for(int i=0;i<count_;i++){
	  //		  if(data_ptr_tmp[i]<thre && data_ptr_tmp[i]>(-thre)){
	  //			data_ptr_tmp[i]=0;
	  //		  }
	  //	  }
	  caffe_gpu_zerout(data_->mutable_gpu_data(),count_,thre);
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <typename Dtype>
void Blob<Dtype>::Disconnect(DisconnectMode mode,Dtype thre, int group) {
	this->Zerout(thre);
  if (NULL == connectivity_.get()) {
    connectivity_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
	if(mode == ELTWISE){
		switch (Caffe::mode()) {
			case Caffe::CPU: {
				  caffe_cpu_if_nonzerout(count_,
						  static_cast<const Dtype*>(data_->cpu_data()),
						  static_cast<Dtype*>(connectivity_->mutable_cpu_data()),
              thre);
				  break;
			}
			case Caffe::GPU: {
#ifndef CPU_ONLY
				  caffe_gpu_if_nonzerout(count_,
						  static_cast<const Dtype*>(data_->gpu_data()),
						  static_cast<Dtype*>(connectivity_->mutable_gpu_data()),
              thre);
#else
			  NO_GPU;
#endif
				break;
			}
			default:
			  LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}else if(mode == GRPWISE){
		CHECK_GE(group,1);
		for (int g = 0; g < group; ++g) {
			caffe_cpu_all_zero_mask(shape_[0]/group,
					count_/shape_[0],
					static_cast<const Dtype*>(data_->cpu_data()) + count_/group * g,
					static_cast<Dtype*>(connectivity_->mutable_cpu_data()) + count_/group * g);
		}
	}

}

template <typename Dtype>
Dtype Blob<Dtype>::GetSparsity(Dtype thre){
  if (4 == num_axes() && shape(0) == shape(1) && (shape(0) == 6 || shape(0) == 8)) {
    // winograd layer
    int N = shape(2);
    int C = shape(3);
    int K = shape(0) - 4 + 1;

    WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
    int M = A->M;

    Dtype *temp = new Dtype[(N*C)*(K*K)];
    caffe_cpu_gemm(
        CblasTrans, CblasTrans,
        N*C, K*K, M*M,
        (Dtype)1, cpu_data(),
        A->getInv()->cpu_data(),
        (Dtype)0, temp);
    caffe_cpu_if_zerout((N*C)*(K*K), temp, temp, thre);
    Dtype sparsity = caffe_cpu_asum((N*C)*(K*K), temp)/(N*C*K*K);
    delete[] temp;
    return sparsity;
  }
  else {
    int zero_num = 0;
    for(int i=0;i<this->count();i++){
      if( this->cpu_data()[i]<=thre && this->cpu_data()[i]>=-thre){
        zero_num++;
      }
    }
    return (Dtype)(zero_num) / (Dtype)(this->count());
  }
}

template <typename Dtype>
Dtype Blob<Dtype>::GetWinogradSparsity(Dtype thre){
  if (4 == num_axes() && shape(2) == shape(3) && (shape(2) == 3 || shape(2) == 5)) {
    int N = shape(0);
    int C = shape(1);
    int K = shape(2);

    WinogradGKronG<Dtype> *A = WinogradGKronG<Dtype>::getInstance(K);
    int M = A->M;

    Dtype *temp = new Dtype[(N*C)*(M*M)];
    caffe_cpu_gemm(
        CblasNoTrans, CblasTrans,
        N*C, M*M, K*K,
        (Dtype)1, cpu_data(),
        A->get()->cpu_data(),
        (Dtype)0, temp);
    caffe_cpu_if_zerout((N*C)*(M*M), temp, temp, thre);
    Dtype sparsity = caffe_cpu_asum((N*C)*(M*M), temp)/(N*C*M*M);
    delete[] temp;
    return sparsity;
  }
  else if (4 == num_axes() && shape(0) == shape(1) && (shape(0) == 6 || shape(0) == 8)) {
    // winograd layer
    int zero_num = 0;
    for(int i=0;i<this->count();i++){
      if( this->cpu_data()[i]<=thre && this->cpu_data()[i]>=-thre){
        zero_num++;
      }
    }
    return (Dtype)(zero_num) / (Dtype)(this->count());
  }
  else {
    return GetSparsity(thre);
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> long Blob<long>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
    return caffe_cpu_asum( prv_data_count(), prv_data());
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> long Blob<long>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
    return caffe_cpu_asum( prv_diff_count(), prv_diff());
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> long Blob<long>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
      data = prv_data();
      sumsq = caffe_cpu_dot(prv_data_count(), data, data);
      break;
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> long Blob<long>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
      diff = prv_diff();
      sumsq = caffe_cpu_dot(prv_diff_count(), diff, diff);
      break;
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<long>::scale_data(long scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
      data = mutable_prv_data();
      caffe_scal(prv_data_count(), scale_factor, data);
      break;
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<long>::scale_diff(long scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::SYNCED_PRV:
  case SyncedMemory::HEAD_AT_PRV:
      diff = mutable_prv_diff();
      caffe_scal(prv_diff_count(), scale_factor, diff);
      break;
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::InitializeConnectivity(Dtype val){
    CHECK(connectivity_);
    caffe_set(count_, val, static_cast<Dtype*>(connectivity_->mutable_cpu_data()));
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
      if (connectivity_.get()) {
        caffe_copy(count_, source.gpu_connectivity(),
                  static_cast<Dtype*>(connectivity_->mutable_gpu_data()));
      }
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
      if (connectivity_.get()) {
        caffe_copy(count_, source.cpu_connectivity(),
                  static_cast<Dtype*>(connectivity_->mutable_cpu_data()));
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template <> void Blob<unsigned int>::ToProto(BlobProto* proto, bool write_diff) const { NOT_IMPLEMENTED; }
template <> void Blob<int>::ToProto(BlobProto* proto, bool write_diff) const { NOT_IMPLEMENTED; }
template <> void Blob<long>::ToProto(BlobProto* proto, bool write_diff) const { NOT_IMPLEMENTED; }
template <> void Blob<size_t>::ToProto(BlobProto* proto, bool write_diff) const { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Snapshot(string filename, bool write_diff) const{
	if(filename.empty()){
		filename = shape_string()+".blob";
	}
	BlobProto proto;
	ToProto(&proto, write_diff);
	WriteProtoToBinaryFile(proto, filename.c_str());
}

template <typename Dtype>
void Blob<Dtype>:: Write1DTensorToNistMMIO(string filename, const Dtype *data_ptr, int I) {
  MM_typecode matcode;
  FILE * fp = fopen(filename.c_str(), "w+");
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);
  mm_set_general(&matcode);

  mm_write_banner(fp, matcode);
  mm_write_mtx_array_size(fp, I, 1);

  /* NOTE: matrix market files stored in column-major order by convention, and
           use 1-based indices, i.e. first element of a vector has index 1, not 0. */
  for (int i=0; i<I; i++) {
    fprintf(fp, "%20.16g\n", (double)(*(data_ptr + i)) );
  }

  fclose(fp);
}

template <typename Dtype>
void Blob<Dtype>:: Write2DTensorToNistMMIO(string filename, const Dtype *data_ptr, int I0, int I1) {
  MM_typecode matcode;
  FILE * fp = fopen(filename.c_str(), "w+");
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);
  mm_set_general(&matcode);

  mm_write_banner(fp, matcode);
  mm_write_mtx_array_size(fp, I0, I1);

  /* NOTE: matrix market files stored in column-major order by convention, and
           use 1-based indices, i.e. first element of a vector has index 1, not 0. */
  for (int j=0; j<I1; j++) {
    for (int i=0; i<I0; i++) {
      fprintf(fp, "%20.16g\n", (double)(*(data_ptr + i * I1 + j)) );
    }
  }

  fclose(fp);
}

template <typename Dtype>
void Blob<Dtype>:: Write4DTensorToNistMMIO(string filename, const Dtype *data_ptr, int I0, int I1, int I2, int I3) {
  MM_typecode matcode;
  FILE * fp = fopen(filename.c_str(), "w+");
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);
  mm_set_general(&matcode);

  mm_write_banner(fp, matcode);
  mm_write_mtx_array_size(fp, I0, I1*I2*I3);

  /* NOTE: matrix market files stored in column-major order by convention, and
           use 1-based indices, i.e. first element of a vector has index 1, not 0. */
  for (int c=0; c<I1; c++) {
    for (int h=0; h<I2; h++) {
      for (int w=0; w<I3; w++) {
        for (int n=0; n<I0; n++) { // for each output channel
          fprintf(fp, "%20.16g\n", (double)(*(data_ptr+((n * I1 + c) * I2 + h) * I3 + w)));
        }
      }
    }
  }

  fclose(fp);
}

template <typename Dtype>
void Blob<Dtype>:: WriteToNistMMIO(string filename) const{
  if(filename.empty()){
    filename = shape_string()+".blob";
  }

  if(num_axes()==4){
    Write4DTensorToNistMMIO(filename, this->cpu_data(), this->shape(0), this->shape(1), this->shape(2), this->shape(3));
  }else if(num_axes()==2){
    Write2DTensorToNistMMIO(filename, this->cpu_data(), this->shape(0), this->shape(1));
  }else if(num_axes()==1) {
    Write1DTensorToNistMMIO(filename, this->cpu_data(), this->shape(0));
  }
  else {
    assert(false);
  }
}

template <typename Dtype>
void Blob<Dtype>:: Write2DTensorToNistMMIOSparse(string filename, const Dtype *data_ptr, int I0, int I1)
{
  MM_typecode matcode;
  FILE * fp = fopen(filename.c_str(), "w+");
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(fp, matcode);

  int nnz = 0;
  for (int i = 0; i < I0*I1; ++i) {
    if (data_ptr[i] != 0) ++nnz;
  }

  mm_write_mtx_crd_size(fp, I0, I1, nnz);

  /* NOTE: matrix market files stored in column-major order by convention, and
           use 1-based indices, i.e. first element of a vector has index 1, not 0. */
  for (int j=0; j<I1; j++) {
    for (int i=0; i<I0; i++) {
      double v = (double)(*(data_ptr + i * I1 + j));
      if (v != 0) fprintf(fp, "%d %d %20.16g\n", i + 1, j + 1, v);
    }
  }
  
  fclose(fp);
}

template <typename Dtype>
void Blob<Dtype>:: Write3DTensorToNistMMIOSparse(string filename, const Dtype *data_ptr, int I0, int I1, int I2)
{
  MM_typecode matcode;
  FILE * fp = fopen(filename.c_str(), "w+");
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(fp, matcode);

  int nnz = 0;
  for (int i = 0; i < I0*I1*I2; ++i) {
    if (data_ptr[i] != 0) ++nnz;
  }

  mm_write_mtx_crd_size(fp, I0, I1*I2, nnz);

  /* NOTE: matrix market files stored in column-major order by convention, and
           use 1-based indices, i.e. first element of a vector has index 1, not 0. */
  for (int h=0; h<I1; h++) {
    for (int w=0; w<I2; w++) {
      for (int n=0; n<I0; n++) { // for each channel
        double v = (double)(*(data_ptr+(n * I1 + h) * I2 + w));
        if (v != 0) {
          fprintf(fp, "%d %d %20.16g\n", n + 1, h*I2 + w + 1, v);
        }
      }
    }
  }

  fclose(fp);
}

template <typename Dtype>
void Blob<Dtype>:: Write4DTensorToNistMMIOSparse(string filename, const Dtype *data_ptr, int I0, int I1, int I2, int I3)
{
  MM_typecode matcode;
  FILE * fp = fopen(filename.c_str(), "w+");
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(fp, matcode);

  int nnz = 0;
  for (int i = 0; i < I0*I1*I2*I3; ++i) {
    if (data_ptr[i] != 0) ++nnz;
  }

  mm_write_mtx_crd_size(fp, I0, I1*I2*I3, nnz);

  /* NOTE: matrix market files stored in column-major order by convention, and
           use 1-based indices, i.e. first element of a vector has index 1, not 0. */
  for (int c=0; c<I1; c++) { // for each input channel
    for (int h=0; h<I2; h++) {
      for (int w=0; w<I3; w++) {
        for (int n=0; n<I0; n++) { // for each output channel
          double v = (double)(*(data_ptr+((n * I1 + c) * I2 + h) * I3 + w));
          if (v != 0) {
            fprintf(fp, "%d %d %20.16g\n", n + 1, (c*I2 + h)*I3 + w + 1, v);
          }
        }
      }
    }
  }
  
  fclose(fp);
}

template <typename Dtype>
void Blob<Dtype>:: WriteToNistMMIOSparse(string filename) const{
	if(filename.empty()){
		filename = shape_string()+".blob";
	}

	if(num_axes()==4){
	  Write4DTensorToNistMMIOSparse(filename, this->cpu_data(), this->shape(0), this->shape(1), this->shape(2), this->shape(3));
	}else if(num_axes()==3){
	  Write3DTensorToNistMMIOSparse(filename, this->cpu_data(), this->shape(0), this->shape(1), this->shape(2));
  }else if(num_axes()==2){
    Write2DTensorToNistMMIOSparse(filename, this->cpu_data(), this->shape(0), this->shape(1));
  }
  else {
    assert(false);
  }
}


INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<size_t>; // Intel caffe
template class Blob<unsigned int>;
template class Blob<long>;

}  // namespace caffe

