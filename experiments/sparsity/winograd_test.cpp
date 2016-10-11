#include <mkl.h>
#include <float.h>

#include "caffe/util/winograd.hpp"

template <typename Dtype>
double get_max_abs_non_diag(Dtype *A, int m, int n)
{
  double maximum = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        maximum = std::max(maximum, fabs(A[i*n + j]));
      }
    }
  }
  return maximum;
}

template <typename Dtype>
double get_max_diag(Dtype *A, int m, int n)
{
  double maximum = DBL_MIN;
  for (int i = 0; i < m; ++i) {
    maximum = std::max(maximum, fabs(A[i*n + i]));
  }
  return maximum;
}

template <typename Dtype>
double get_min_diag(Dtype *A, int m, int n)
{
  double minimum = DBL_MAX;
  for (int i = 0; i < m; ++i) {
    minimum = std::min(minimum, fabs(A[i*n + i]));
  }
  return minimum;
}

int main()
{
  {
    const float *GGTInv = get_GGTInv_4x4_3x3<float>();
    const float *GGT = get_G_kron_G_4x4_3x3<float>()->cpu_data();

    float temp[(3*3)*(3*3)];

    float alpha = 1, beta = 0;
    int lda = 6*6, ldb = 3*3, ldc = 3*3;
    int M = 3*3, N = 3*3, K = 6*6;
    cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      M, N, K,
      alpha, GGTInv, lda,
      GGT, ldb,
      beta, temp, ldc);

    printf(
      "max_abs_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 3*3, 3*3), 
      get_min_diag(temp, 3*3, 3*3),
      get_max_diag(temp, 3*3, 3*3));
    //print_matrix(temp, 3*3, 3*3);
  }

  {
    const double *GGTInv = get_GGTInv_4x4_3x3<double>();
    const double *GGT = get_G_kron_G_4x4_3x3<double>()->cpu_data();

    double temp[(3*3)*(3*3)];

    double alpha = 1, beta = 0;
    int lda = 6*6, ldb = 3*3, ldc = 3*3;
    int M = 3*3, N = 3*3, K = 6*6;
    cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      M, N, K,
      alpha, GGTInv, lda,
      GGT, ldb,
      beta, temp, ldc);

    printf(
      "max_abs_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 3*3, 3*3), 
      get_min_diag(temp, 3*3, 3*3),
      get_max_diag(temp, 3*3, 3*3));
    //print_matrix(temp, 3*3, 3*3);
  } 

  {
    const float *GGTInv = get_GGTInv_4x4_5x5<float>();
    const float *GGT = get_G_kron_G_4x4_5x5<float>()->cpu_data();

    float temp[(5*5)*(5*5)];

    float alpha = 1, beta = 0;
    int lda = 8*8, ldb = 5*5, ldc = 5*5;
    int M = 5*5, N = 5*5, K = 8*8;
    cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      M, N, K,
      alpha, GGTInv, lda,
      GGT, ldb,
      beta, temp, ldc);

    printf(
      "max_abs_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 5*5, 5*5), 
      get_min_diag(temp, 5*5, 5*5),
      get_max_diag(temp, 5*5, 5*5));
    //print_matrix(temp, 5*5, 5*5);
  }

  {
    const double *GGTInv = get_GGTInv_4x4_5x5<double>();
    const double *GGT = get_G_kron_G_4x4_5x5<double>()->cpu_data();

    double temp[(5*5)*(5*5)];

    double alpha = 1, beta = 0;
    int lda = 8*8, ldb = 5*5, ldc = 5*5;
    int M = 5*5, N = 5*5, K = 8*8;
    cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      M, N, K,
      alpha, GGTInv, lda,
      GGT, ldb,
      beta, temp, ldc);

    printf(
      "max_abs_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 5*5, 5*5), 
      get_min_diag(temp, 5*5, 5*5),
      get_max_diag(temp, 5*5, 5*5));
    //print_matrix(temp, 5*5, 5*5);
  }

  return 0;
}
