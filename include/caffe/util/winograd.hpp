#ifndef CAFFE_UTIL_WINOGRAD_H_
#define CAFFE_UTIL_WINOGRAD_H_

#include "caffe/blob.hpp"

/**
 * compute Kronecker product of in1 and in2, where in1 is a m by n matrix and in2 is a p by q matrix
 *
 * @params out an (m*p) by (n*q) matrix stored in row major
 * @params in1 an m by n matrix stored in row major
 * @params in2 an p by q matrix stored in row major
 */
template <typename Dtype>
void kronecker_product(Dtype *out, const double *in1, const double *in2, int m, int n, int p, int q)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < p; ++k) {
        for (int l = 0; l < q; ++l) {
          out[(p*i + k)*n*q + q*j + l] = in1[n*i + j]*in2[k*q + l];
            /* compute in double precision and then convert it back to Dtype for accuracy */
        }
      }
    }
  }
}

/**
 * C = A^T*B
 *
 * @params C an m by n matrix stored in row major
 * @params A an k by m matrix stored in row major
 * @params B an k by n matrix stored in row major
 */
template<typename Dtype>
void atb(Dtype *C, const Dtype *A, const Dtype *B, int m, int n, int k)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Dtype sum = 0;
      for (int l = 0; l < k; ++l) {
        sum += A[l*m + i]*B[n*l + j];
      }
      C[n*i + j] = sum;
    }
  }
}

/**
 * C = A*B^T
 *
 * @params C an m by n matrix stored in row major
 * @params A an m by k matrix stored in row major
 * @params B an n by k matrix stored in row major
 */
template<typename Dtype>
void abt(Dtype *C, const Dtype *A, const Dtype *B, int m, int n, int k)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Dtype sum = 0;
      for (int l = 0; l < k; ++l) {
        sum += A[k*i + l]*B[k*j + l];
      }
      C[n*i + j] = sum;
    }
  }
}

template<typename Dtype>
void transpose(Dtype *AT, const Dtype *A, int m, int n)
{
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      AT[i*m + j] = A[j*n + i];
    }
  }
}

template<typename Dtype>
void transpose_to_float(float *AT, const Dtype *A, int m, int n)
{
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      AT[i*m + j] = A[j*n + i];
    }
  }
}

template<typename Dtype>
void print_matrix(const Dtype *A, int m, int n, int lda)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      fprintf(stderr, "%g ", A[i*lda + j]);
    }
    fprintf(stderr, "\n");
  }
}

template<typename Dtype>
void print_matrix(std::ostringstream& str, const Dtype *A, int m, int n, int lda)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      str << A[i*lda + j] << " ";
    }
    str << std::endl;
  }
}

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_transpose_of(
  const Dtype *data, int m, int n)
{
  std::vector<int> shape;
  shape.push_back(n);
  shape.push_back(m);
  boost::shared_ptr<caffe::Blob<Dtype> > blob(new caffe::Blob<Dtype>(shape));

  transpose(blob->mutable_cpu_data(), data, m, n);

  return blob;
}

template<typename Dtype>
void print_matrix(const Dtype *A, int m, int n)
{
  return print_matrix(A, m, n, n);
}

template<typename Dtype>
void print_matrix(std::ostringstream& str, const Dtype *A, int m, int n)
{
  return print_matrix(str, A, m, n, n);
}

template <typename Dtype>
void potrf(const char *uplo, const int *n, Dtype *a, const int *lda, int *info);

template<>
void potrf(const char *uplo, const int *n, float *a, const int *lda, int *info){
  spotrf(uplo, n, a, lda, info);
}

template<>
void potrf(const char *uplo, const int *n, double *a, const int *lda, int *info){
  dpotrf(uplo, n, a, lda, info);
}

template <typename Dtype>
void potrs(const char *uplo, const int *n, const int *nrhs, const Dtype *a, const int *lda, Dtype *b, const int *ldb, int *info);

template<>
void potrs(const char *uplo, const int *n, const int *nrhs, const float *a, const int *lda, float *b, const int *ldb, int *info)
{
  spotrs(uplo, n, nrhs, a, lda, b, ldb, info);
}

template<>
void potrs(const char *uplo, const int *n, const int *nrhs, const double *a, const int *lda, double *b, const int *ldb, int *info)
{
  dpotrs(uplo, n, nrhs, a, lda, b, ldb, info);
}

template<int K>
struct WinogradParameters
{
};

template<>
struct WinogradParameters<3>
{
  static const int M = 6;
  static const int N = 3;

  static const double *getG() {
    static const double G[M*N] = {
       1./ 4.,       0,      0,
      -1./ 6., -1./ 6., -1./6.,
      -1./ 6.,  1./ 6., -1./6.,
       1./24.,  1./12.,  1./6.,
       1./24., -1./12.,  1./6.,
            0,       0,      1,
    };
    return G;
  }
};

template<>
struct WinogradParameters<5>
{
  static const int M = 8;
  static const int N = 5;

  static const double *getG() {
    static const double G[M*N] = {
            1,       0,       0,       0,        0,
      -2./ 9., -2./ 9., -2./ 9., -2./ 9., -2./  9.,
      -2./ 9.,  2./ 9., -2./ 9.,  2./ 9., -2./  9.,
       1./90.,  1./45.,  2./45.,  4./45.,  8./ 45.,
       1./90., -1./45.,  2./45., -4./45.,  8./ 45.,
       4./45.,  2./45.,  1./45.,  1./90.,  1./180.,
       4./45., -2./45.,  1./45., -1./90.,  1./180.,
            0,       0,       0,       0,        1,
    };
    return G;
  }
};

template<typename Dtype>
class WinogradGKronG
{
private :
  const double *G;

  WinogradGKronG(int K) : K(K) {
    if (3 == K) {
      M = WinogradParameters<3>::M;
      N = WinogradParameters<3>::N;
      G = WinogradParameters<3>::getG();
    }
    else if (5 == K) {
      M = WinogradParameters<5>::M;
      N = WinogradParameters<5>::N;
      G = WinogradParameters<5>::getG();
    }
    else {
      assert(false);
      M = 0;
      N = 0;
      G = NULL;
    }
  }

  boost::shared_ptr<caffe::Blob<Dtype> > GKronG;
  boost::shared_ptr<caffe::Blob<Dtype> > transpose;
  boost::shared_ptr<caffe::Blob<Dtype> > transposeNormalizedWithInv;
  boost::shared_ptr<caffe::Blob<Dtype> > normOfRows;

  boost::shared_ptr<Dtype> inv;
  boost::shared_ptr<caffe::Blob<Dtype> > invTranspose;
  boost::shared_ptr<caffe::Blob<Dtype> > normOfInvCols;

public :
  const int K;
  int M, N;

  static WinogradGKronG *getInstance(int K) {
    static WinogradGKronG *instances[16] = { NULL };
    if (3 == K) {
      if (instances[3] == NULL) {
        instances[3] = new WinogradGKronG(3);
      }
      return instances[3];
    }
    else if (5 == K) {
      if (instances[5] == NULL) {
        instances[5] = new WinogradGKronG(5);
      }
      return instances[5];
    }
    else {
      assert(false);
      return NULL;
    }
  }

  const boost::shared_ptr<caffe::Blob<Dtype> > get() {
    if (NULL == GKronG.get()) {
      std::vector<int> shape;
      shape.push_back(M*M);
      shape.push_back(N*N);
      GKronG = boost::shared_ptr<caffe::Blob<Dtype> >(new caffe::Blob<Dtype>(shape));

      kronecker_product(GKronG->mutable_cpu_data(), G, G, M, N, M, N);
    }

    return GKronG;
  }

  const boost::shared_ptr<caffe::Blob<Dtype> > getTranspose() {
    if (NULL == transpose.get()) {
      transpose = get_transpose_of(get()->cpu_data(), M*M, N*N);
    }

    return transpose;
  }
//
//  template <typename Dtype>
//  static const boost::shared_ptr<caffe::Blob<Dtype> > getNormalizedTranspose() {
//    static bool initialized = false;
//    static boost::shared_ptr<caffe::Blob<float> > GKronGTranspose;
//
//    if (!initialized) {
//      const double *normOfRows = getNormOfRows<double>()->cpu_data();
//      const double *GKronG = get<double>()->cpu_data();
//      Dtype temp[(M*M)*(N*N)];
//
//      for (int i = 0; i < M*M; ++i) {
//        for (int j = 0; j < N*N; ++j) {
//          temp[i*N*N + j] = GKronG[i*N*N + j]/normOfRows[i];
//        }
//      }
//
//      GKronGTranspose = get_transpose_of(temp, M*M, N*N);
//      initialized = true;
//    }
//
//    return GKronGTranspose;
//  }
//
  const boost::shared_ptr<caffe::Blob<Dtype> > getTransposeNormalizedWithInv() {
    if (NULL == transposeNormalizedWithInv.get()) {
      const double *normOfCols = WinogradGKronG<double>::getInstance(K)->getNormOfInvCols()->cpu_data();
      const double *GKronG = WinogradGKronG<double>::getInstance(K)->get()->cpu_data();
      Dtype temp[(M*M)*(N*N)];

      for (int i = 0; i < M*M; ++i) {
        for (int j = 0; j < N*N; ++j) {
          temp[i*N*N + j] = GKronG[i*N*N + j]*normOfCols[i];
        }
      }

      transposeNormalizedWithInv = get_transpose_of(temp, M*M, N*N);
    }

    return transposeNormalizedWithInv;
  }

  const boost::shared_ptr<caffe::Blob<Dtype> > getNormOfRows() {
    if (NULL == normOfRows.get()) {
      std::vector<int> shape;
      shape.push_back(M*M);
      normOfRows = boost::shared_ptr<caffe::Blob<Dtype> >(new caffe::Blob<Dtype>(shape));
      Dtype *row_wise_l2norm_data = normOfRows->mutable_cpu_data();

      const double *G_kron_G = get()->cpu_data();
      for (int i = 0; i < M*M; ++i) {
        double sum = 0;
        for (int j = 0; j < N*N; ++j) {
          sum += G_kron_G[i*N*N + j]*G_kron_G[i*N*N + j];
        }
        row_wise_l2norm_data[i] = sqrt(sum);
      }
    }

    return normOfRows;
  }

//  static double getFrobNorm() {
//    static bool initialized = false;
//    static double ret = 0;
//    if (!initialized) {
//      double sum = 0;
//      double *A = get<double>()->cpu_data();
//      for (int i = 0; i < get<double>()->count(); ++i) {
//        sum += ((double)A[i])*((double)A[i]);
//      }
//      ret = sqrt((double)sum)
//      LOG(INFO) << "||G \\kron G|| " << ret;
//      initialized = true;
//    }
//    return ret;
//  }

  const Dtype *getInv() {
    if (NULL == inv.get()) {
      inv.reset(new Dtype[(N*N)*(M*M)]);

      double S[N*N];
      double U[(M*M)*(N*N)], VT[(N*N)*(N*N)];
      double superb[N*N - 1];

      double A_temp[(M*M)*(N*N)];
      memcpy(A_temp, WinogradGKronG<double>::getInstance(K)->get()->cpu_data(), sizeof(A_temp));
      // NOTE: A_temp will be overwritten by LAPACKE_dgesvd
      MKL_INT info = LAPACKE_dgesvd(
        LAPACK_ROW_MAJOR, 'S', 'A',
        M*M, N*N,
        A_temp, N*N,
        S,
        U, N*N,
        VT, N*N,
        superb);
      if (info > 0) {
        LOG(FATAL) << "SVD failed to converge with return value " << info;
      }

      double S_pinv[(N*N)*(N*N)];
      memset(S_pinv, 0, sizeof(S_pinv));
      for (int i = 0; i < N*N; ++i) {
        S_pinv[i*N*N + i] = 1/S[i];
      }

      double V_times_S_pinv[(N*N)*(N*N)];

      double alpha = 1, beta = 0;
      int m = N*N, n = N*N, k = N*N;
      int lda = k, ldb = n, ldc = n;
      cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        m, n, k,
        alpha, VT, lda,
        S_pinv, ldb,
        beta, V_times_S_pinv, ldc);

      double A_inv_temp[(N*N)*(M*M)];
      m = N*N, n = M*M, k = N*N;
      lda = k, ldb = k, ldc = n;
      cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        m, n, k,
        alpha, V_times_S_pinv, lda,
        U, ldb,
        beta, A_inv_temp, ldc);

      for (int i = 0; i < (N*N)*(M*M); ++i) {
        inv.get()[i] = A_inv_temp[i];
      }
    }

    return inv.get();
  }

  const boost::shared_ptr<caffe::Blob<Dtype> > getTransposeOfInv() {
    if (NULL == invTranspose.get()) {
      invTranspose = get_transpose_of(getInv(), N*N, M*M);
    }

    return invTranspose;
  }

  const boost::shared_ptr<caffe::Blob<Dtype> > getNormOfInvCols() {
    if (NULL == normOfInvCols.get()) {
      std::vector<int> shape;
      shape.push_back(M*M);
      normOfInvCols = boost::shared_ptr<caffe::Blob<Dtype> >(new caffe::Blob<Dtype>(shape));
      Dtype *col_wise_l2norm_data = normOfInvCols->mutable_cpu_data();

      const double *GGTInv = WinogradGKronG<double>::getInstance(K)->getInv();
      for (int i = 0; i < M*M; ++i) {
        double sum = 0;
        for (int j = 0; j < N*N; ++j) {
          sum += GGTInv[j*M*M + i]*GGTInv[j*M*M + i];
        }
        col_wise_l2norm_data[i] = sqrt(sum);
      }
    }

    return normOfInvCols;
  }

  /**
   * @param weight_input weights in time domain that is going to be updated according
   *                     to sparsity pattern in mask
   * @param mask 0 element means it's thresholded
   * @param impose_factor 0 means we're not imposing sparsity at all.
   */
  void imposeSparsity(Dtype *weight_inout, const Dtype *mask, double impose_factor) {
    const double *A = WinogradGKronG<double>::getInstance(K)->get()->cpu_data();

    // [impose_factor*A_I; eye]
    double A_temp[(M*M + N*N)*(N*N)];
    int zero_cnt = 0;
    for (int i = 0; i < M*M; ++i) {
      if (mask[i] == 0) {
        for (int j = 0; j < N*N; ++j) {
          A_temp[zero_cnt*N*N + j] = impose_factor*A[i*N*N + j];
        }
        ++zero_cnt;
      }
    }
    memset(A_temp + zero_cnt*N*N, 0, sizeof(A_temp[0])*(N*N)*(N*N));
    for (int i = 0; i < N*N; ++i) {
      A_temp[(zero_cnt + i)*(N*N) + i] = 1;
    }

    // [zeros; W_t]
    double b[zero_cnt + (N*N)];
    memset(b, 0, sizeof(b[0])*zero_cnt);
    for (int i = 0; i < N*N; ++i) {
      b[zero_cnt + i] = weight_inout[i];
    }

    lapack_int info = LAPACKE_dgels(
        LAPACK_ROW_MAJOR, 'N',
        zero_cnt + N*N, N*N,
        1,
        A_temp, N*N,
        b, 1);
    if (info != 0) {
      LOG(FATAL) << "dgels failed with return value " << info;
    }

    for (int i = 0; i < N*N; ++i) {
      weight_inout[i] = b[i];
    }
  }
}; // WinogradGKronG

#endif
