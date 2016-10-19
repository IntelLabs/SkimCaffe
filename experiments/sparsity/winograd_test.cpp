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
double get_frob_norm_non_diag(Dtype *A, int m, int n)
{
  double sum = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        sum += A[i*n + j]*A[i*n + j];
      }
    }
  }
  return sqrt(sum);
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
    WinogradGKronG<float> *A = WinogradGKronG<float>::getInstance(3);

    const float *GGTInv = A->getInv();
    const float *GGT = A->get()->cpu_data();

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
      "max_abs_non_diag = %g frob_norm_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 3*3, 3*3), 
      get_frob_norm_non_diag(temp, 3*3, 3*3),
      get_min_diag(temp, 3*3, 3*3),
      get_max_diag(temp, 3*3, 3*3));
    //print_matrix(temp, 3*3, 3*3);
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(3);

    const double *GGTInv = A->getInv();
    const double *GGT = A->get()->cpu_data();

    for (int i = 0; i < 6*6; ++i) {
      printf("%g ", A->getNormOfRows()->cpu_data()[i]);
    }
    printf("\n");

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
      "max_abs_non_diag = %g frob_norm_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 3*3, 3*3), 
      get_frob_norm_non_diag(temp, 3*3, 3*3),
      get_min_diag(temp, 3*3, 3*3),
      get_max_diag(temp, 3*3, 3*3));
    //print_matrix(temp, 3*3, 3*3);
  } 

  {
    WinogradGKronG<float> *A = WinogradGKronG<float>::getInstance(5);

    const float *GGTInv = A->getInv();
    const float *GGT = A->get()->cpu_data();

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
      "max_abs_non_diag = %g frob_norm_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 5*5, 5*5), 
      get_frob_norm_non_diag(temp, 5*5, 5*5),
      get_min_diag(temp, 5*5, 5*5),
      get_max_diag(temp, 5*5, 5*5));
    //print_matrix(temp, 5*5, 5*5);
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(5);

    const double *GGTInv = A->getInv();
    const double *GGT = A->get()->cpu_data();

    for (int i = 0; i < 8*8; ++i) {
      printf("%g ", A->getNormOfRows()->cpu_data()[i]);
    }
    printf("\n");

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
      "max_abs_non_diag = %g frob_norm_non_diag = %g min_diag = %g max_diag = %g\n",
      get_max_abs_non_diag(temp, 5*5, 5*5), 
      get_frob_norm_non_diag(temp, 5*5, 5*5),
      get_min_diag(temp, 5*5, 5*5),
      get_max_diag(temp, 5*5, 5*5));
    //print_matrix(temp, 5*5, 5*5);
  }

  {
    // Julia code
    // G1=[[1/4 0 0]; [-1/6 -1/6 -1/6]; [-1/6 1/6 -1/6]; [1/24 1/12 1/6]; [1/24 -1/12 1/6]; [0 0 1]]
    // A1=kron(G1,G1)
    // C1=A1[[1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], :]
    // V1=svd(C1)[3]
    // W-V1*V1'*W
    //
    // C1=A1[[2, 9, 17, 22, 25, 26], :]
    // V1=svd(C1)[3]
    // W-V1*V1'*W

    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(3);

    double mask[36] = { // 36 = 6*6
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0,
        1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };

    {
      double weight[9] = {
          1, 2, 3,
          4, 5, 6,
          7, 8, 9,
      };

      A->imposeSparsity(weight, mask);

      fprintf(stderr, "W'\n"); print_matrix(weight, 3, 3);
    }

    {
      for (int i = 0; i < 36; ++i) {
        mask[i] = (mask[i] == 0) ? 1 : 0;
      }

      double weight[9] = {
          1, 2, 3,
          4, 5, 6,
          7, 8, 9,
      };

      A->imposeSparsity(weight, mask);

      fprintf(stderr, "W'\n"); print_matrix(weight, 3, 3);
    }
  }

  {
    // Julia code
    // G1=[[1/4 0 0]; [-1/6 -1/6 -1/6]; [-1/6 1/6 -1/6]; [1/24 1/12 1/6]; [1/24 -1/12 1/6]; [0 0 1]]
    // A1=kron(G1,G1)
    // C1=A1[[15], :]
    // V1=svd(C1)[3]
    // W-V1*V1'*W
    //
    // C1=A1[[2, 9, 17, 22, 25, 26], :]
    // V1=svd(C1)[3]
    // W-V1*V1'*W

    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(3);

    double mask[36] = { // 36 = 6*6
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };

    {
      for (int i = 0; i < 36; ++i) {
        mask[i] = (mask[i] == 0) ? 1 : 0;
      }

      double weight[9] = {
          -0.23985,  0.32602, 0.481601,
          0.235871, 0.291445, 0.676182,
          0.579017, 0.289492, 0.398896,
      };

      A->imposeSparsity(weight, mask);

      fprintf(stderr, "W'\n"); print_matrix(weight, 3, 3);
    }
  }

  return 0;
}
