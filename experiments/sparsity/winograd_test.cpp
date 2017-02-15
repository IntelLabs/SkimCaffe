#include <float.h>
#include <mkl.h>

#include "caffe/util/winograd.hpp"
#include "winograd_test.h"

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
  if (false)
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

  if (false)
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

  if (false)
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

  if (false)
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

  if (false)
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

    if (false)
    {
      double weight[9] = {
          1, 2, 3,
          4, 5, 6,
          7, 8, 9,
      };

      A->imposeSparsity(weight, mask, 1);

      fprintf(stderr, "W'\n"); print_matrix(weight, 3, 3);

//      double *weight_gpu, *mask_gpu;
//      cudaError_t cudaStat = cudaMalloc((void **)&weight_gpu, sizeof(weight));
//      assert(cudaSuccess == cudaStat);
//      cudaStat = cudaMalloc((void **)&mask_gpu, sizeof(mask));
//      assert(cudaSuccess == cudaStat);

//          for (int i = 0; i < 36; ++i) {
//            mask[i] = (mask[i] == 0) ? 1 : 0;
//          }

//      imposeSparsityGPU(weight, mask, A->get()->gpu_data(), A->M, A->N);
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

      A->imposeSparsity(weight, mask, 1);

//      fprintf(stderr, "W'\n"); print_matrix(weight, 3, 3);

//      imposeSparsityGPU(weight, mask, A->get()->gpu_data(), A->M, A->N);
    }
  }

  // If there're more than N*N zeros, then V is full rank -> V*V^T is identity matrix

  if (false)
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

//    double mask[36] = { // 36 = 6*6
//        0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0,
//        0, 0, 1, 0, 0, 0,
//        0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0,
//    };

    double mask[36] = { // 36 = 6*6
        1, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 1,
    };

    {
//      for (int i = 0; i < 36; ++i) {
//        mask[i] = (mask[i] == 0) ? 1 : 0;
//      }

      double weight[9] = {
          -0.23985,  0.32602, 0.481601,
          0.235871, 0.291445, 0.676182,
          0.579017, 0.289492, 0.398896,
      };

      A->imposeSparsity(weight, mask, 1);

      fprintf(stderr, "W'\n"); print_matrix(weight, 3, 3);
    }
  }

  if (false)
  {

//    W2=[-0.0157232 0.207913 -0.00964544 0.129141 -0.0579629
//           0.143075 -0.035662 0.116638 -0.0962759 0.161726
//           -0.115694 -0.247545 0.00910714 0.412703 0.470526
//           0.182568 -0.0347781 -0.161185 0.114429 0.202669
//           0.211487 0.217568 -0.246375 -0.275411 0.211334
//           ];
//    W2=reshape(W2', 25);
//    C2=A2[[21, 31, 36, 38, 39, 43, 45, 46, 47, 52, 53, 54], :];
//    V2=svd(C2)[3];
//    W2-V2*V2'*W2

    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(5);

    double mask[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 1, 0, 1, 0, 0, 1,
        1, 1, 0, 1, 0, 0, 0, 1,
        1, 1, 1, 0, 0, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
    };

    double weight[25] = {
      -0.0157232, 0.207913, -0.00964544, 0.129141, -0.0579629,
      0.143075, -0.035662, 0.116638, -0.0962759, 0.161726,
      -0.115694, -0.247545, 0.00910714, 0.412703, 0.470526,
      0.182568, -0.0347781, -0.161185, 0.114429, 0.202669,
      0.211487, 0.217568, -0.246375, -0.275411, 0.211334,
    };

    A->imposeSparsity(weight, mask, 1);

//    print_matrix(weight, 5, 5);

//    imposeSparsityGPU(weight, mask, A->get()->gpu_data(), A->M, A->N);
  }

  if (false)
  {
//    W1=[0.00407881 0.00373635 -0.000966112 -0.00248292 -0.0658406 0.0494906 0.0137974 -0.00570734 -0.0224007]'
//    C1=A1[[1,2,3,4,5,8,10,14,26],:]
//    V1=(svd(C1)[3])[:,1:6]
//    W1-V1*V1'W1
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(3);

    double weight[3*3] = {
        -0.23985,  0.32602, 0.481601, 0.235871, 0.291445, 0.676182, 0.579017, 0.289492, 0.398896,
    };

    int indices[] = { 15 };
    double mask[A->M*A->M];
    for (int i = 0; i < sizeof(mask)/sizeof(mask[0]); ++i) {
      mask[i] = 1;
    }
    for (int i = 0; i < sizeof(indices)/sizeof(indices[0]); ++i) {
      mask[indices[i] - 1] = 0;
    }

    double weight_temp[A->N*A->N];

    memcpy(weight_temp, weight, sizeof(weight));
    A->imposeSparsity(weight_temp, mask, 1);
    print_matrix(weight_temp, A->N, A->N);

    memcpy(weight_temp, weight, sizeof(weight));
    A->imposeSparsity(weight_temp, mask, 100);
    print_matrix(weight_temp, A->N, A->N);
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(3);

    double weight[3*3] = {
        0.0154869, 0.0118699, 0.00100924, -0.0625147, 0.0720939, -0.000762911, -0.00286087, -0.047404, 0.0281393,
    };

    int indices[] = { 3, 5, 8, 11, 14, 16, 20, 21, 28, };
    double mask[A->M*A->M];
    for (int i = 0; i < sizeof(mask)/sizeof(mask[0]); ++i) {
      mask[i] = 1;
    }
    for (int i = 0; i < sizeof(indices)/sizeof(indices[0]); ++i) {
      mask[indices[i] - 1] = 0;
    }

    double weight_temp[A->N*A->N];

    memcpy(weight_temp, weight, sizeof(weight));
    A->imposeSparsity(weight_temp, mask, 1);
    print_matrix(weight_temp, A->N, A->N);

    memcpy(weight_temp, weight, sizeof(weight));
    A->imposeSparsity(weight_temp, mask, 100);
    print_matrix(weight_temp, A->N, A->N);
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(5);

    double weight[5*5] = {
      -0.0157232, 0.207913, -0.00964544, 0.129141, -0.0579629, 0.143075, -0.035662, 0.116638, -0.0962759, 0.161726, -0.115694, -0.247545, 0.00910714, 0.412703, 0.470526, 0.182568, -0.0347781, -0.161185, 0.114429, 0.202669, 0.211487, 0.217568, -0.246375, -0.275411, 0.211334,
    };

    int indices[] = { 21, 31, 36, 38, 39, 43, 45, 46, 47, 52, 53, 54, };
    double mask[A->M*A->M];
    for (int i = 0; i < sizeof(mask)/sizeof(mask[0]); ++i) {
      mask[i] = 1;
    }
    for (int i = 0; i < sizeof(indices)/sizeof(indices[0]); ++i) {
      mask[indices[i] - 1] = 0;
    }

    double weight_temp[A->N*A->N];

    memcpy(weight_temp, weight, sizeof(weight));
    A->imposeSparsity(weight_temp, mask, 1);
    print_matrix(weight_temp, A->N, A->N);

    memcpy(weight_temp, weight, sizeof(weight));
    A->imposeSparsity(weight_temp, mask, 100);
    print_matrix(weight_temp, A->N, A->N);
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(3);

    float weight_cpu[2][3*3] = {
        { -0.23985,  0.32602, 0.481601, 0.235871, 0.291445, 0.676182, 0.579017, 0.289492, 0.398896, },
        { 0.0154869, 0.0118699, 0.00100924, -0.0625147, 0.0720939, -0.000762911, -0.00286087, -0.047404, 0.0281393, },
    };
    int repeat = sizeof(weight_cpu)/sizeof(weight_cpu[0]);

    std::vector<int> shape;
    shape.push_back(repeat);
    shape.push_back(sizeof(weight_cpu[0])/sizeof(weight_cpu[0][0]));
    caffe::Blob<float> weight(shape);

    shape.clear();
    shape.push_back(repeat);
    shape.push_back(A->M*A->M + A->N*A->N);
    caffe::Blob<double> weight_temp(shape);

    shape.clear();
    shape.push_back(repeat);
    caffe::Blob<long> weight_temp_ptr(shape);

    shape.clear();
    shape.push_back(repeat);
    shape.push_back(A->M*A->M + A->N*A->N);
    shape.push_back(A->N*A->N);
    caffe::Blob<double> A_temp(shape);

    shape.clear();
    shape.push_back(repeat);
    caffe::Blob<long> A_temp_ptr(shape);

    for (int i = 0; i < repeat; ++i) {
      ((double **)weight_temp_ptr.mutable_cpu_data())[i] = weight_temp.mutable_gpu_data() + i*(A->M*A->M + A->N*A->N);
      ((double **)A_temp_ptr.mutable_cpu_data())[i] = A_temp.mutable_gpu_data() + i*(A->M*A->M + A->N*A->N)*(A->N*A->N);

      for (int j = 0; j < A->M*A->M; ++j) {
        weight_temp.mutable_cpu_data()[i*(A->M*A->M + A->N*A->N) + j] = 0;
      }
      for (int j = A->M*A->M; j < A->M*A->M + A->N*A->N; ++j) {
        for (int k = 0; k < A->N*A->N; ++k) {
          A_temp.mutable_cpu_data()[(i*(A->N*A->N) + k)*(A->M*A->M + A->N*A->N) + j] = 0;
        }
        A_temp.mutable_cpu_data()[(i*(A->N*A->N) + j - A->M*A->M)*(A->M*A->M + A->N*A->N) + j] = 1;
      }
    }

    int indices[2][6*6] = {
      { 15, 0, },
      { 3, 5, 8, 11, 14, 16, 20, 21, 28, 0, },
    };

    shape.clear();
    shape.push_back(A->M*A->M*repeat);
    caffe::Blob<float> mask(shape);
    for (int i = 0; i < shape[0]; ++i) {
      mask.mutable_cpu_data()[i] = 1;
    }
    for (int i = 0; i < repeat; ++i) {
      for (int j = 0; j < sizeof(indices[0])/sizeof(indices[0][0]); ++j) {
        if (indices[i][j] == 0) break;
        mask.mutable_cpu_data()[A->M*A->M*i + indices[i][j] - 1] = 0;
      }
    }

    for (int i = 0; i < repeat; ++i) {
      memcpy(weight.mutable_cpu_data() + i*sizeof(weight_cpu[0])/sizeof(weight_cpu[0][0]), weight_cpu[i], sizeof(weight_cpu[0]));
    }

    caffe::caffe_gpu_impose_sparsity(
      weight.mutable_gpu_data(), weight_temp.mutable_gpu_data(), (double **)weight_temp_ptr.mutable_gpu_data(),
      A->get()->gpu_data(), A_temp.mutable_gpu_data(), (double **)A_temp_ptr.mutable_gpu_data(),
      mask.mutable_gpu_data(), 1, A->M, A->N, repeat);

    for (int i = 0; i < repeat; ++i) {
      print_matrix(weight.cpu_data() + i*A->N*A->N, A->N, A->N);
      fprintf(stderr, "\n");
    }
  }

  {
    WinogradGKronG<double> *A = WinogradGKronG<double>::getInstance(5);

    float weight_cpu[1][5*5] = {
      { -0.0157232, 0.207913, -0.00964544, 0.129141, -0.0579629, 0.143075, -0.035662, 0.116638, -0.0962759, 0.161726, -0.115694, -0.247545, 0.00910714, 0.412703, 0.470526, 0.182568, -0.0347781, -0.161185, 0.114429, 0.202669, 0.211487, 0.217568, -0.246375, -0.275411, 0.211334, },
    };
    int repeat = sizeof(weight_cpu)/sizeof(weight_cpu[0]);

    std::vector<int> shape;
    shape.push_back(repeat);
    shape.push_back(sizeof(weight_cpu[0])/sizeof(weight_cpu[0][0]));
    caffe::Blob<float> weight(shape);

    shape.clear();
    shape.push_back(repeat);
    shape.push_back(A->M*A->M + A->N*A->N);
    caffe::Blob<double> weight_temp(shape);

    shape.clear();
    shape.push_back(repeat);
    caffe::Blob<long> weight_temp_ptr(shape);

    shape.clear();
    shape.push_back(repeat);
    shape.push_back(A->M*A->M + A->N*A->N);
    shape.push_back(A->N*A->N);
    caffe::Blob<double> A_temp(shape);

    shape.clear();
    shape.push_back(repeat);
    caffe::Blob<long> A_temp_ptr(shape);

    for (int i = 0; i < repeat; ++i) {
      ((double **)weight_temp_ptr.mutable_cpu_data())[i] = weight_temp.mutable_gpu_data() + i*(A->M*A->M + A->N*A->N);
      ((double **)A_temp_ptr.mutable_cpu_data())[i] = A_temp.mutable_gpu_data() + i*(A->M*A->M + A->N*A->N)*(A->N*A->N);

      for (int j = 0; j < A->M*A->M; ++j) {
        weight_temp.mutable_cpu_data()[i*(A->M*A->M + A->N*A->N) + j] = 0;
      }
      for (int j = A->M*A->M; j < A->M*A->M + A->N*A->N; ++j) {
        for (int k = 0; k < A->N*A->N; ++k) {
          A_temp.mutable_cpu_data()[(i*(A->N*A->N) + k)*(A->M*A->M + A->N*A->N) + j] = 0;
        }
        A_temp.mutable_cpu_data()[(i*(A->N*A->N) + j - A->M*A->M)*(A->M*A->M + A->N*A->N) + j] = 1;
      }
    }

    int indices[1][8*8] = {
      { 21, 31, 36, 38, 39, 43, 45, 46, 47, 52, 53, 54, },
    };

    shape.clear();
    shape.push_back(A->M*A->M*repeat);
    caffe::Blob<float> mask(shape);
    for (int i = 0; i < shape[0]; ++i) {
      mask.mutable_cpu_data()[i] = 1;
    }
    for (int i = 0; i < repeat; ++i) {
      for (int j = 0; j < sizeof(indices[0])/sizeof(indices[0][0]); ++j) {
        if (indices[i][j] == 0) break;
        mask.mutable_cpu_data()[A->M*A->M*i + indices[i][j] - 1] = 0;
      }
    }

    for (int i = 0; i < repeat; ++i) {
      memcpy(weight.mutable_cpu_data() + i*sizeof(weight_cpu[0])/sizeof(weight_cpu[0][0]), weight_cpu[i], sizeof(weight_cpu[0]));
    }

    caffe::caffe_gpu_impose_sparsity(
      weight.mutable_gpu_data(), weight_temp.mutable_gpu_data(), (double **)weight_temp_ptr.mutable_gpu_data(),
      A->get()->gpu_data(), A_temp.mutable_gpu_data(), (double **)A_temp_ptr.mutable_gpu_data(),
      mask.mutable_gpu_data(), 100, A->M, A->N, repeat);

    for (int i = 0; i < repeat; ++i) {
      print_matrix(weight.cpu_data() + i*A->N*A->N, A->N, A->N);
      fprintf(stderr, "\n");
    }
  }

  return 0;
}
