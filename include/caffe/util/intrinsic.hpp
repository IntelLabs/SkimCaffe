#ifndef _CAFFE_UTIL_INTRINSIC_HPP_
#define _CAFFE_UTIL_INTRINSIC_HPP_

#ifdef __AVX512F__

#define VLEN (16)
#define SIMDFPTYPE __m512

#define _MM_SET1(a) _mm512_set1_ps(a)
#define _MM_FMADD(a, b, c) _mm512_fmadd_ps(a, b, c)
#define _MM_ADD(a, b) _mm512_add_ps(a, b)
#define _MM_MAX(a, b) _mm512_max_ps(a, b)
#define _MM_SETZERO() _mm512_setzero_ps()

#define _MM_LOAD(a) _mm512_load_ps(a)
#define _MM_STORE(a, b) _mm512_store_ps(a, b)

#elif __AVX2__

#define VLEN (8)
#define SIMDFPTYPE __m256

#define _MM_SET1(a) _mm256_set1_ps(a)
#define _MM_FMADD(a, b, c) _mm256_fmadd_ps(a, b, c)
#define _MM_ADD(a, b) _mm256_add_ps(a, b)
#define _MM_MAX(a, b) _mm256_max_ps(a, b)
#define _MM_SETZERO() _mm256_setzero_ps()

#define _MM_LOAD(a) _mm256_load_ps(a)
#define _MM_STORE(a, b) _mm256_store_ps(a, b)

#else

#define VLEN (4)
#define SIMDFPTYPE __m128

#define _MM_SET1(a) _mm_set1_ps(a)
#define _MM_FMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define _MM_ADD(a, b) _mm_add_ps(a, b)
#define _MM_MAX(a, b) _mm_max_ps(a, b)
#define _MM_SETZERO() _mm_setzero_ps()

#define _MM_LOAD(a) _mm_load_ps(a)
#define _MM_STORE(a, b) _mm_store_ps(a, b)

#endif

#endif // _CAFFE_UTIL_INTRINSIC_HPP_
