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
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm.h>
#include "libxsmm/src/libxsmm_main.h"
#include "caffe/util/libxsmm_spmv.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_SPMDM_MALLOC_INTRINSIC
#endif
#if defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC)
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) _mm_malloc(SIZE, ALIGNMENT)
# define LIBXSMM_SPMDM_FREE(BUFFER) _mm_free((void*)(BUFFER))
#else
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) libxsmm_aligned_malloc(SIZE, -(ALIGNMENT))
# define LIBXSMM_SPMDM_FREE(BUFFER) libxsmm_free(BUFFER)
#endif

/* Enable/disable specific code paths */
#if !defined(LIBXSMM_SPMDM_AVX512_CORE)
# define LIBXSMM_SPMDM_AVX512_CORE
#endif
#if !defined(LIBXSMM_SPMDM_AVX2)
# define LIBXSMM_SPMDM_AVX2
#endif


#if !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];
#endif


/* function pointer for the CPUID-dispatched implementation */
void (*internal_spmv_compute_fp32_thread)(const libxsmm_spmv_handle*, char,
  const float*, libxsmm_CSR_sparseslice*, const float*, const float*, float*, int, int, int);


LIBXSMM_INLINE LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX)
void internal_spmdm_init_shufmask_avx()
{
#if !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_LEGACY) \
  && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  unsigned int i, j, c, last_bit;
  LIBXSMM_ALIGNED(int temp_shufmasks[8], 64);
  LIBXSMM_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  int cnt;
  for (i = 0; i < 256; i++) {
    cnt = 0;
    j = i;
    for (c = 0; c < 8; c++) temp_shufmasks[c] = 0;
    for (c = 0; c < 16; c++) temp_shufmasks2[c] = 0;
    while ( j) {
      last_bit = LIBXSMM_INTRINSICS_BITSCANFWD(j);
      temp_shufmasks[cnt] = last_bit;
      temp_shufmasks2[cnt] = (uint16_t)last_bit;
      j &= (~(1<<last_bit));
      cnt++;
    }
    internal_spmdm_shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    internal_spmdm_shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
#endif
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmv_allocate_csr_a(libxsmm_spmv_handle* handle, libxsmm_CSR_sparseslice** libxsmm_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  size_t sz_block = ((handle->bm + 1)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(float) + sizeof(libxsmm_CSR_sparseslice));
  size_t sz_all_blocks = sz_block * handle->mb * handle->kb;

  char * memory_block = (char *)LIBXSMM_SPMDM_MALLOC( sz_all_blocks, 2097152);
  char * memory_head  = memory_block;

  libxsmm_CSR_sparseslice* libxsmm_output_csr_a = (libxsmm_CSR_sparseslice*)(memory_head);
  memory_head += handle->mb * handle->kb * sizeof(libxsmm_CSR_sparseslice);

  for (kb = 0; kb < k_blocks; kb++) {
    for (mb = 0; mb < m_blocks; mb++) {
      int i = kb*m_blocks + mb;
      libxsmm_output_csr_a[i].rowidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm + 1)*sizeof(uint16_t);
      libxsmm_output_csr_a[i].colidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(uint16_t);
      libxsmm_output_csr_a[i].values = (float*)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(float);
    }
  }
  assert(memory_head == (memory_block + sz_all_blocks));
  *libxsmm_output_csr = libxsmm_output_csr_a;
  handle->base_ptr_scratch_A = memory_block;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmv_deallocate_csr_a(libxsmm_spmv_handle* handle)
{
  LIBXSMM_SPMDM_FREE(handle->base_ptr_scratch_A);
  handle->base_ptr_scratch_A= NULL;
}


LIBXSMM_API_DEFINITION void libxsmm_spmv_destroy(libxsmm_spmv_handle* handle)
{
  internal_spmv_deallocate_csr_a(handle);
}


LIBXSMM_API_DEFINITION
void libxsmm_spmv_createSparseSlice_fp32_thread(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
  libxsmm_spmdm_handle spmdm_handle;
  spmdm_handle.m = handle->m;
  spmdm_handle.k = handle->k;
  spmdm_handle.bm = handle->bm;
  spmdm_handle.bk = handle->bk;
  spmdm_handle.mb = handle->mb;
  spmdm_handle.kb = handle->kb;
  spmdm_handle.datatype = handle->datatype;
  spmdm_handle.base_ptr_scratch_A = handle->base_ptr_scratch_A;

//  libxsmm_spmdm_createSparseSlice_fp32_thread(
//      &spmdm_handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);

  {
    const libxsmm_spmdm_handle *handle = &spmdm_handle;
# include "../src/libxsmm/src/libxsmm_spmdm_begin_avx2.h"
# include "../src/libxsmm/src/template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "../src/libxsmm/src/libxsmm_spmdm_end.h"
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void internal_spmv_compute_fp32_thread_sw(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float* alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float* B,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
  const int m_blocks = handle->mb;
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int k_block_size = handle->bk;
  int mb = block_id;

  int m_overall_start = mb*m_block_size;
  int m_overall_end   = (mb + 1)*m_block_size;
  int m, kb;

  int k_overall_start, k_overall_end, num_k;

  LIBXSMM_UNUSED(nthreads);
  LIBXSMM_UNUSED(transA);
  LIBXSMM_UNUSED(tid);

  if (m_overall_end > handle->m) m_overall_end = handle->m;

  for (kb = 0; kb < k_blocks; kb++) {
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;

    if (1.f == *alpha) {
      if (1.f == *beta || kb > 0) {
        for (m = m_overall_start; m < m_overall_end; m++, m_local++) {
          int start_j, end_j, j;
          float sum;

          if (m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

          start_j = slice.rowidx[m_local];
          end_j   = slice.rowidx[m_local + 1];

          sum = C[m];
          for (j = start_j; j < end_j; j++) {
            sum += B[k_overall_start + slice.colidx[j]]*slice.values[j];
          }
          C[m] = sum;
        }
      }
      else if (0.f == *beta) {
//#define DBG_SPMDM
#ifdef DBG_SPMDM
        int ROW_TO_DEBUG = 0;
#endif
        for (m = m_overall_start; m < m_overall_end; m++, m_local++) {
          int start_j, end_j, j;
          float sum;

          if (m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

          start_j = slice.rowidx[m_local];
          end_j   = slice.rowidx[m_local + 1];

          sum = 0;
          for (j = start_j; j < end_j; j++) {
            sum += B[k_overall_start + slice.colidx[j]]*slice.values[j];
#ifdef DBG_SPMDM
            if (ROW_TO_DEBUG == m)
              printf("%g*%d:%g + ", slice.values[j], k_overall_start + slice.colidx[j], B[k_overall_start + slice.colidx[j]]);
#endif
          }
          C[m] = sum;
#ifdef DBG_SPMDM
          if (ROW_TO_DEBUG == m) printf(" = %g kb = %d\n", sum, kb);
#undef DBG_SPMDM
#endif
        }
      }
      else {
        for (m = m_overall_start; m < m_overall_end; m++, m_local++) {
          int start_j, end_j, j;
          float sum;

          if (m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

          start_j = slice.rowidx[m_local];
          end_j   = slice.rowidx[m_local + 1];

          sum = *beta*C[m];
          for (j = start_j; j < end_j; j++) {
            sum += B[k_overall_start + slice.colidx[j]]*slice.values[j];
          }
          C[m] = sum;
        }
      }
    }
    else {
      if (1.f == *beta || kb > 0) {
        for (m = m_overall_start; m < m_overall_end; m++, m_local++) {
          int start_j, end_j, j;
          float sum;

          if (m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

          start_j = slice.rowidx[m_local];
          end_j   = slice.rowidx[m_local + 1];

          sum = C[m];
          for (j = start_j; j < end_j; j++) {
            sum += B[k_overall_start + slice.colidx[j]]*slice.values[j];
          }
          C[m] = *alpha*sum;
        }
      }
      else {
        for (m = m_overall_start; m < m_overall_end; m++, m_local++) {
          int start_j, end_j, j;
          float sum;

          if (m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

          start_j = slice.rowidx[m_local];
          end_j   = slice.rowidx[m_local + 1];

          sum = 0;
          for (j = start_j; j < end_j; j++) {
            sum += B[k_overall_start + slice.colidx[j]]*slice.values[j];
          }
          C[m] = *alpha*sum + *beta*C[m];
        }
      }
    }
  } /* kb */
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
void internal_spmv_compute_fp32_thread_avx2(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float* alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float* B,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
  internal_spmv_compute_fp32_thread_sw(handle, transA, alpha, A_sparse, B, beta, C, block_id, tid, nthreads);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void internal_spmv_compute_fp32_thread_avx512_core(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float* alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float* B,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
  internal_spmv_compute_fp32_thread_sw(handle, transA, alpha, A_sparse, B, beta, C, block_id, tid, nthreads);
}


LIBXSMM_API_DEFINITION
void libxsmm_spmv_compute_fp32_thread(
  const libxsmm_spmv_handle* handle,
  char transA,
  const float* alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float* B,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmv_compute_fp32_thread_avx512_core(handle, transA, alpha, A_sparse, B, beta, C, block_id, tid, nthreads);
#elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmv_compute_fp32_thread_avx2(handle, transA, alpha, A_sparse, B, beta, C, block_id, tid, nthreads);
#else /* pointer based function call */
  assert(0 != internal_spmv_compute_fp32_thread);
  internal_spmv_compute_fp32_thread(handle, transA, alpha, A_sparse, B, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_spmv_init(int M, int K, int max_threads,
  libxsmm_spmv_handle* handle, libxsmm_CSR_sparseslice** libxsmm_output_csr)
{
  /* initialize internal library structures */
  LIBXSMM_INIT

  handle->m  = M;
  handle->k  = K;

  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid) {
    internal_spmv_compute_fp32_thread = internal_spmv_compute_fp32_thread_avx512_core;
  }
  else if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid) {
    internal_spmv_compute_fp32_thread = internal_spmv_compute_fp32_thread_avx2;
  }
  else {
    internal_spmv_compute_fp32_thread = internal_spmv_compute_fp32_thread_sw;
  }

  handle->mb = max_threads;
  handle->bm = (handle->m + handle->mb - 1) / handle->mb;

  handle->bk = 128;
  handle->kb = (handle->k + handle->bk - 1) / handle->bk;

  /* This is temporary space needed; allocate for each different size of A */
  internal_spmv_allocate_csr_a(handle, libxsmm_output_csr);

  /* Initialize shuffle masks for the computation */
  if (LIBXSMM_X86_AVX <= libxsmm_target_archid) {
    internal_spmdm_init_shufmask_avx();
  }

  /* post-conditions */
  assert(0 != internal_spmv_compute_fp32_thread);
}

