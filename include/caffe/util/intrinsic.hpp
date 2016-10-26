#ifndef _CAFFE_UTIL_INTRINSIC_HPP_
#define _CAFFE_UTIL_INTRINSIC_HPP_

#include <immintrin.h>

#ifdef __AVX512F__

#define VLEN (16)
#define SIMDFPTYPE __m512
#define SIMDITYPE __m512i

#define _MM_SET1(a) _mm512_set1_ps(a)
#define _MM_SET1_EPI32(a) _mm512_set1_epi32(a)
#define _MM_SETZERO() _mm512_setzero_ps()

#define _MM_ADD(a, b) _mm512_add_ps(a, b)
#define _MM_SUB(a, b) _mm512_sub_ps(a, b)
#define _MM_MUL(a, b) _mm512_mul_ps(a, b)
#define _MM_POW(a, b) _mm512_pow_ps(a, b)
#define _MM_FMADD(a, b, c) _mm512_fmadd_ps(a, b, c)

#define _MM_MAX(a, b) _mm512_max_ps(a, b)

#define _MM_LOAD(a) _mm512_load_ps(a)
#define _MM_LOADU(a) _mm512_loadu_ps(a)
#define _MM_LOAD_SI(a) _mm512_load_si512(a)

#define _MM_STORE(a, b) _mm512_store_ps(a, b)
#define _MM_STOREU(a, b) _mm512_storeu_ps(a, b)
#define _MM_MASK_STORE(a, mask, b) _mm512_mask_storeu_ps(a, mask, b)

#elif __AVX2__

#define VLEN (8)
#define SIMDFPTYPE __m256
#define SIMDITYPE __m256i

#define _MM_SET1(a) _mm256_set1_ps(a)
#define _MM_SET1_EPI32(a) _mm256_set1_epi32(a)
#define _MM_SETZERO() _mm256_setzero_ps()

#define _MM_ADD(a, b) _mm256_add_ps(a, b)
#define _MM_SUB(a, b) _mm256_sub_ps(a, b)
#define _MM_MUL(a, b) _mm256_mul_ps(a, b)
#define _MM_FMADD(a, b, c) _mm256_fmadd_ps(a, b, c)
#define _MM_POW(a, b) _mm256_pow_ps(a, b)

#define _MM_MAX(a, b) _mm256_max_ps(a, b)

#define _MM_LOAD(a) _mm256_load_ps(a)
#define _MM_LOADU(a) _mm256_loadu_ps(a)
#define _MM_LOAD_SI(a) _mm256_load_si256(a)

#define _MM_STORE(a, b) _mm256_store_ps(a, b)
#define _MM_STOREU(a, b) _mm256_storeu_ps(a, b)
#define _MM_MASK_STORE(a, mask, b) _mm256_maskstore_ps(a, mask, b)

#else

#define VLEN (4)
#define SIMDFPTYPE __m128
#define SIMDITYPE __m128i

#define _MM_SET1(a) _mm_set1_ps(a)
#define _MM_SET1_EPI32(a) _mm_set1_epi32(a)
#define _MM_SETZERO() _mm_setzero_ps()

#define _MM_ADD(a, b) _mm_add_ps(a, b)
#define _MM_SUB(a, b) _mm_sub_ps(a, b)
#define _MM_MUL(a, b) _mm_mul_ps(a, b)
#define _MM_FMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define _MM_POW(a, b) _mm_pow_ps(a, b)

#define _MM_MAX(a, b) _mm_max_ps(a, b)

#define _MM_LOAD(a) _mm_load_ps(a)
#define _MM_LOADU(a) _mm_loadu_ps(a)
#define _MM_LOAD_SI(a) _mm_load_si128(a)

#define _MM_STORE(a, b) _mm_store_ps(a, b)
#define _MM_STOREU(a, b) _mm_storeu_ps(a, b)
#define _MM_MASK_STORE(a, mask, b) _mm_maskstore_ps(a, mask, b)

#endif

#endif // _CAFFE_UTIL_INTRINSIC_HPP_
