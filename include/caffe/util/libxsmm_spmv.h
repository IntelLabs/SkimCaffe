/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Jongsoo Park (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_SPMV_H
#define LIBXSMM_SPMV_H

#include <libxsmm_macros.h>
#include <libxsmm_spmdm.h>

typedef struct libxsmm_spmv_handle {
  /* The following are the matrix multiply dimensions: A (sparse): m X k, x (dense): k X 1, Output y (dense): m X 1 */
  /* In general, the convention is calling the dimension of A as m x n but we use m x k convention instead to be consistent with spmdm */
  int m;
  int k;
  /* The block sizes for A, B and C. */
  /* Here we fix A to be divided into 128 X 128 blocks, B/C to be 128 X 48 for HSW/BDW and 128 X 96 for SKX */
  int bm;
  int bk;
  /* The number of blocks for the m, n and k dimensions */
  int mb;
  int kb;
  libxsmm_spmdm_datatype datatype;
  char * base_ptr_scratch_A;
} libxsmm_spmv_handle;

LIBXSMM_API void libxsmm_spmv_init(
  int M, int K,
  int max_threads,
  libxsmm_spmv_handle* handle,
  libxsmm_CSR_sparseslice** libxsmm_output_csr);

LIBXSMM_API void libxsmm_spmv_destroy(
  libxsmm_spmv_handle * handle);

/* Don't need libxsmm_spmv_get_num_*_blocks functions like spmdm because we assume a simple
 * 1-D blocking along rows of A.
 */
/*LIBXSMM_API int libxsmm_spmv_get_num_createSparseSlice_blocks(
  const libxsmm_spmv_handle* handle);

LIBXSMM_API int libxsmm_spmv_get_num_compute_blocks(
  const libxsmm_spmv_handle* handle);*/

/** This converts a dense representation of the sparse matrix to 2D array of sparse slices. */
LIBXSMM_API void libxsmm_spmv_createSparseSlice_fp32_thread(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads);

/*LIBXSMM_API void libxsmm_spmv_createSparseSlice_bfloat16_thread(
  const libxsmm_spmv_handle* handle,
  char transA,
  const uint16_t * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads);*/

/** NOTE: This code currently ignores alpha input to the matrix multiply */
LIBXSMM_API void libxsmm_spmv_compute_fp32_thread(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float *B,
  const float *beta,
  float* C,
  int block_id,
  int tid, int nthreads);

/** NOTE: This code currently ignores alpha input to the matrix multiply */
/*LIBXSMM_API void libxsmm_spmv_compute_bfloat16_thread(
  const libxsmm_spmv_handle* handle,
  char transA,
  const uint16_t *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const uint16_t *B,
  const uint16_t *beta,
  float* C,
  int block_id,
  int tid, int nthreads);*/

#endif /*LIBXSMM_SPMV_H*/
