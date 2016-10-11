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
void print_matrix(const Dtype *A, int m, int n)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%g ", A[i*n + j]);
    }
    printf("\n");
  }
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

const double G_4x4_3x3[6*3] = {
   1./ 4.,       0,      0,
  -1./ 6., -1./ 6., -1./6.,
  -1./ 6.,  1./ 6., -1./6.,
   1./24.,  1./12.,  1./6.,
   1./24., -1./12.,  1./6.,
        0,       0,      1,
};

const double G_4x4_5x5[8*5] = {
         1,       0,       0,       0,        0,
  -2.f/ 9., -2./ 9., -2./ 9., -2./ 9., -2./  9.,
  -2.f/ 9.,  2./ 9., -2./ 9.,  2./ 9., -2./  9.,
   1.f/90.,  1./45.,  2./45.,  4./45.,  8./ 45.,
   1.f/90., -1./45.,  2./45., -4./45.,  8./ 45.,
   4.f/45.,  2./45.,  1./45.,  1./90.,  1./180.,
   4.f/45., -2./45.,  1./45., -1./90.,  1./180.,
         0,       0,       0,       0,        1,
};

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_G_kron_G_4x4_3x3()
{
  const int M = 6;
  const int N = 3;

  static bool initialized = false;
  static boost::shared_ptr<caffe::Blob<Dtype> > G_kron_G;

  if (!initialized) {
    std::vector<int> shape;
    shape.push_back(M*M);
    shape.push_back(N*N);
    G_kron_G = boost::shared_ptr<caffe::Blob<Dtype> >(new caffe::Blob<Dtype>(shape));

    kronecker_product(G_kron_G->mutable_cpu_data(), G_4x4_3x3, G_4x4_3x3, M, N, M, N);
    initialized = true;
  }

  return G_kron_G;
}

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_G_kron_G_4x4_5x5()
{
  const int M = 8;
  const int N = 5;

  static bool initialized = false;
  static boost::shared_ptr<caffe::Blob<Dtype> > G_kron_G;

  if (!initialized) {
    std::vector<int> shape;
    shape.push_back(M*M);
    shape.push_back(N*N);
    G_kron_G = boost::shared_ptr<caffe::Blob<Dtype> >(new caffe::Blob<Dtype>(shape));

    kronecker_product(G_kron_G->mutable_cpu_data(), G_4x4_5x5, G_4x4_5x5, M, N, M, N);
    initialized = true;
  }

  return G_kron_G;
}

template <typename Dtype>
double get_l2_norm_of(const caffe::Blob<Dtype> *A)
{
  double sum = 0;
  const Dtype *Aptr = A->cpu_data();
  for (int i = 0; i < A->count(); ++i) {
    sum += ((double)Aptr[i])*((double)Aptr[i]);
  }
  return sqrt((double)sum);
}

double get_Frob_norm_of_G_kron_G_4x4_3x3()
{
  static bool initialized = false;
  static double ret = 0;
  if (!initialized) {
    ret = get_l2_norm_of(get_G_kron_G_4x4_3x3<float>().get());
    LOG(INFO) << "||G \\kron G|| " << ret;
    initialized = true;
  }
  return ret;
}

double get_Frob_norm_of_G_kron_G_4x4_5x5()
{
  static bool initialized = false;
  static double ret = 0;
  if (!initialized) {
    ret = get_l2_norm_of(get_G_kron_G_4x4_5x5<float>().get());
    LOG(INFO) << "||G \\kron G|| " << ret;
    initialized = true;
  }
  return ret;
}

static bool NORMALIZE_WINOGRAD_FACTORS = true;

template<typename Dtype>
Dtype get_threshold(int k)
{
  return double(ZEROUT_THRESHOLD) * (NORMALIZE_WINOGRAD_FACTORS ? (k == 3 ? (get_Frob_norm_of_G_kron_G_4x4_3x3()*k*k/6/6) : (get_Frob_norm_of_G_kron_G_4x4_5x5()*k*k/8/8)) : 1);
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

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_G_kron_G_transpose_4x4_3x3();

template <>
const boost::shared_ptr<caffe::Blob<double> > get_G_kron_G_transpose_4x4_3x3()
{
  NOT_IMPLEMENTED;
  return boost::shared_ptr<caffe::Blob<double> >();
}

template <>
const boost::shared_ptr<caffe::Blob<float> > get_G_kron_G_transpose_4x4_3x3()
{
  const int M = 6;
  const int N = 3;

  static bool initialized = false;
  static boost::shared_ptr<caffe::Blob<float> > G_kron_G_transpose;

  if (!initialized) {
    G_kron_G_transpose =
      get_transpose_of(get_G_kron_G_4x4_3x3<float>()->cpu_data(), M*M, N*N);
    initialized = true;
  }

  return G_kron_G_transpose;
}

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_G_kron_G_transpose_4x4_5x5();

template <>
const boost::shared_ptr<caffe::Blob<double> > get_G_kron_G_transpose_4x4_5x5()
{
  NOT_IMPLEMENTED;
  return boost::shared_ptr<caffe::Blob<double> >();
}

template <>
const boost::shared_ptr<caffe::Blob<float> > get_G_kron_G_transpose_4x4_5x5()
{
  const int M = 8;
  const int N = 5;

  static bool initialized = false;
  static boost::shared_ptr<caffe::Blob<float> > G_kron_G_transpose;

  if (!initialized) {
    G_kron_G_transpose =
      get_transpose_of(get_G_kron_G_4x4_5x5<float>()->cpu_data(), M*M, N*N);
    initialized = true;
  }

  return G_kron_G_transpose;
}

template <typename Dtype>
const Dtype *get_GGTInv_4x4_3x3();

template <>
const double *get_GGTInv_4x4_3x3()
{
  const int M = 6;
  const int N = 3;

  static bool initialized = false;
  static double GGTInv[(N*N)*(M*M)];

  if (!initialized) {
    // A = (G \kron G^T)
    const double *A = get_G_kron_G_4x4_3x3<double>()->cpu_data();

    // A^T * A
    double ATA[(N*N)*(N*N)];
    atb(ATA, A, A, N*N, N*N, M*M);

    // Cholesky factorization to compute (A^T * A)^-1
    char uplo = 'L';
    int order = N*N;
    int lda = N*N;
    int info;
    potrf(&uplo, &order, ATA, &lda, &info);
    if (info) {
      LOG(FATAL) << "potrf returned " << info;
    }

    // Solve (A^T * A)^-1 * A^T
    double A_temp[(M*M)*(N*N)];
    memcpy(A_temp, A, sizeof(A_temp));
    int nrhs = M*M;
    int ldb = N*N;
    potrs(&uplo, &order, &nrhs, ATA, &lda, A_temp, &ldb, &info);
      // potrs expects column-major so passing A_temp (initialized as A)
      // in row-major effectively passes A^T
    if (info) {
      LOG(FATAL) << "potrs returned " << info;
    }
    transpose(GGTInv, A_temp, M*M, N*N);

    printf("GGTInv\n"); print_matrix(GGTInv, N*N, M*M); printf("\n");

    initialized = true;
  }

  return GGTInv;
}

template <>
const float *get_GGTInv_4x4_3x3()
{
  const int M = 6;
  const int N = 3;

  static bool initialized = false;
  static float GGTInv[(N*N)*(M*M)];

  if (!initialized) {
    // A = (G \kron G^T)
    const double *A = get_G_kron_G_4x4_3x3<double>()->cpu_data();

    // A^T * A
    double ATA[(N*N)*(N*N)];
    atb(ATA, A, A, N*N, N*N, M*M);

    // Cholesky factorization to compute (A^T * A)^-1
    char uplo = 'L';
    int order = N*N;
    int lda = N*N;
    int info;
    potrf(&uplo, &order, ATA, &lda, &info);
    if (info) {
      LOG(FATAL) << "potrf returned " << info;
    }

    // Solve (A^T * A)^-1 * A^T
    double A_temp[(M*M)*(N*N)];
    memcpy(A_temp, A, sizeof(A_temp));
    int nrhs = M*M;
    int ldb = N*N;
    potrs(&uplo, &order, &nrhs, ATA, &lda, A_temp, &ldb, &info);
      // potrs expects column-major so passing A_temp (initialized as A)
      // in row-major effectively passes A^T
    if (info) {
      LOG(FATAL) << "potrs returned " << info;
    }
    transpose_to_float(GGTInv, A_temp, M*M, N*N);

    printf("GGTInv\n"); print_matrix(GGTInv, N*N, M*M); printf("\n");

    initialized = true;
  }

  return GGTInv;
}

template <typename Dtype>
const Dtype *get_GGTInv_4x4_5x5();

template <>
const double *get_GGTInv_4x4_5x5()
{
  const int M = 8;
  const int N = 5;

  static bool initialized = false;
  static double GGTInv[(N*N)*(M*M)];

  if (!initialized) {
    // A = (G \kron G^T)
    const double *A = get_G_kron_G_4x4_5x5<double>()->cpu_data();

    // A^T * A
    double ATA[(N*N)*(N*N)];
    atb(ATA, A, A, N*N, N*N, M*M);

    // Cholesky factorization to compute (A^T * A)^-1
    char uplo = 'L';
    int order = N*N;
    int lda = N*N;
    int info;
    potrf(&uplo, &order, ATA, &lda, &info);
    if (info) {
      LOG(FATAL) << "potrf returned " << info;
    }

    // Solve (A^T * A)^-1 * A^T
    double A_temp[(M*M)*(N*N)];
    memcpy(A_temp, A, sizeof(A_temp));
    int nrhs = M*M;
    int ldb = N*N;
    potrs(&uplo, &order, &nrhs, ATA, &lda, A_temp, &ldb, &info);
      // potrs expects column-major so passing A_temp (initialized as A)
      // in row-major effectively passes A^T
    if (info) {
      LOG(FATAL) << "potrs returned " << info;
    }
    transpose(GGTInv, A_temp, M*M, N*N);

    printf("GGTInv\n"); print_matrix(GGTInv, N*N, M*M); printf("\n");

    initialized = true;
  }

  return GGTInv;
}

template <>
const float *get_GGTInv_4x4_5x5()
{
  const int M = 8;
  const int N = 5;

  static bool initialized = false;
  static float GGTInv[(N*N)*(M*M)];

  if (!initialized) {
    // A = (G \kron G^T)
    const double *A = get_G_kron_G_4x4_5x5<double>()->cpu_data();

    // A^T * A
    double ATA[(N*N)*(N*N)];
    atb(ATA, A, A, N*N, N*N, M*M);

    // Cholesky factorization to compute (A^T * A)^-1
    char uplo = 'L';
    int order = N*N;
    int lda = N*N;
    int info;
    potrf(&uplo, &order, ATA, &lda, &info);
    if (info) {
      LOG(FATAL) << "potrf returned " << info;
    }

    // Solve (A^T * A)^-1 * A^T
    double A_temp[(M*M)*(N*N)];
    memcpy(A_temp, A, sizeof(A_temp));
    int nrhs = M*M;
    int ldb = N*N;
    potrs(&uplo, &order, &nrhs, ATA, &lda, A_temp, &ldb, &info);
      // potrs expects column-major so passing A_temp (initialized as A)
      // in row-major effectively passes A^T
    if (info) {
      LOG(FATAL) << "potrs returned " << info;
    }
    transpose_to_float(GGTInv, A_temp, M*M, N*N);

    printf("GGTInv\n"); print_matrix(GGTInv, N*N, M*M); printf("\n");

    initialized = true;
  }

  return GGTInv;
}

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_GGTInv_transpose_4x4_3x3();

template <>
const boost::shared_ptr<caffe::Blob<double> > get_GGTInv_transpose_4x4_3x3()
{
  NOT_IMPLEMENTED;
  return boost::shared_ptr<caffe::Blob<double> >();
}

template <>
const boost::shared_ptr<caffe::Blob<float> > get_GGTInv_transpose_4x4_3x3()
{
  const int M = 6;
  const int N = 3;

  static bool initialized = false;
  static boost::shared_ptr<caffe::Blob<float> > GGTInv_transpose;

  if (!initialized) {
    GGTInv_transpose =
      get_transpose_of(get_GGTInv_4x4_3x3<float>(), N*N, M*M);
    initialized = true;
  }

  return GGTInv_transpose;
}

template <typename Dtype>
const boost::shared_ptr<caffe::Blob<Dtype> > get_GGTInv_transpose_4x4_5x5();

template <>
const boost::shared_ptr<caffe::Blob<double> > get_GGTInv_transpose_4x4_5x5()
{
  NOT_IMPLEMENTED;
  return boost::shared_ptr<caffe::Blob<double> >();
}

template <>
const boost::shared_ptr<caffe::Blob<float> > get_GGTInv_transpose_4x4_5x5()
{
  const int M = 8;
  const int N = 5;

  static bool initialized = false;
  static boost::shared_ptr<caffe::Blob<float> > GGTInv_transpose;

  if (!initialized) {
    GGTInv_transpose =
      get_transpose_of(get_GGTInv_4x4_5x5<float>(), N*N, M*M);
    initialized = true;
  }

  return GGTInv_transpose;
}

#endif
