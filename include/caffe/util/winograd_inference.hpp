
#ifndef CAFFE_UTIL_WINOGRAD_INFERENCE_HPP_
#define CAFFE_UTIL_WINOGRAD_INFERENCE_HPP_

//winograd for cpu inference

#include "caffe/blob.hpp"

#define DEBUG_WINOGRAD 0

#if DEBUG_WINOGRAD
	#include <cassert>
#endif

namespace WINOGRAD_INFERENCE {

	const enum WINOGRAD_MATRIX {
		WINOGRAD_A = 0,
		WINOGRAD_B,
		WINOGRAD_G,
	};
	const enum WINOGRAD_ALG {
		WT_8X8_F_6X6_3X3 = 0,
		WT_6X6_F_4X4_3X3,
		WT_8X8_F_4X4_5X5,
	};

	const int WINOGRAD_MATRIX_NUM = 3;
	const int WINOGRAD_ALG_NUM = 3;

	template<WINOGRAD_ALG alg>
	struct WinogradParameters{};

	/**
	* compute Kronecker product of in1 and in2, where in1 is a m by n matrix and in2 is a p by q matrix
	*
	* @params out an (m*p) by (n*q) matrix stored in row major
	* @params in1 an m by n matrix stored in row major
	* @params in2 an p by q matrix stored in row major
	*/
	void kronecker_product(float *out, const float *in1, const float *in2, int m, int n, int p, int q);

	//singleton, precomputation before inference  
	void winograd2D_initialize();

	template<>
	struct WinogradParameters<WT_6X6_F_4X4_3X3>
	{
		// wt6x6, F(4x4,3x3)
	private:
		static const int O = 4;
		static const int K = 3;
		static const int T = O + K - 1;

		static const float *getG() {
			static const float G[T*K] = {
				1. / 4.,       0,      0,
				-1. / 6., -1. / 6., -1. / 6.,
				-1. / 6.,  1. / 6., -1. / 6.,
				1. / 24.,  1. / 12.,  1. / 6.,
				1. / 24., -1. / 12.,  1. / 6.,
				0,       0,      1,
			};
			return G;
		}

		static const float *getA() {
			static const float A[T*O] = {
				1,  0, 0,  0,
				1,  1, 1,  1,
				1, -1, 1, -1,
				1,  2, 4,  8,
				1, -2, 4, -8,
				0,  0, 0,  1,
			};
			return A;
		}

		static const float *getB() {
			static const float B[T*T] = {
				4,  0,  0,  0,  0,  0,
				0, -4,  4, -2,  2,  4,
				-5, -4, -4, -1, -1,  0,
				0,  1, -1,  2, -2, -5,
				1,  1,  1,  1,  1,  0,
				0,  0,  0,  0,  0,  1,
			};
			return B;
		};

	public:
		static const float *get(WINOGRAD_MATRIX mat, int &row, int& col) {

#if DEBUG_WINOGRAD
			assert(mat >= WINOGRAD_A && mat <= WINOGRAD_G);
#endif
			switch (mat) {

			case WINOGRAD_A: row = T; col = O; return getA();
			case WINOGRAD_B: row = T; col = T; return getB();
			case WINOGRAD_G: row = T; col = K; return getG();

			}

		}

	};

	template<>
	struct WinogradParameters<WT_8X8_F_6X6_3X3>
	{

	private:

		// wt8x8, F(6x6,3x3)

		static const int O = 6;
		static const int K = 3;
		static const int T = O + K - 1;

	public:
		static const float *get(WINOGRAD_MATRIX mat, int &row, int& col) {

#if DEBUG_WINOGRAD
			assert(mat >= WINOGRAD_A && mat <= WINOGRAD_G);
#endif
			switch (mat) {

			case WINOGRAD_A: row = T; col = O; return getA();
			case WINOGRAD_B: row = T; col = T; return getB();
			case WINOGRAD_G: row = T; col = K; return getG();

			}

		}

	private:
		static const float *getG() {
			static const float G[T*K] = {
				1.f,   0.f   ,  0.f  ,
				-2.f / 9 , -2.f / 9  , -2.f / 9,
				-2.f / 9 ,  2.f / 9  , -2.f / 9,
				1.f / 90 , 1.f / 45  , 2.f / 45,
				1.f / 90 , -1.f / 45 , 2.f / 45,
				32.f / 45,  16.f / 45, 8.f / 45,
				32.f / 45, -16.f / 45, 8.f / 45,
				0.f   ,   0.f   ,  1.f  ,
			};
			return G;
		}

		static const float *getA() {
			static const float A[T*(T - K + 1)] = {
				1 * 1.f,           0 * 1.f,           0 * 1.f,           0 * 1.f,           0 * 1.f,           0 * 1.f,
				1 * 1.f,           1 * 1.f,           1 * 1.f,           1 * 1.f,           1 * 1.f,           1 * 1.f,
				1 * 1.f,          -1 * 1.f,           1 * 1.f,          -1 * 1.f,           1 * 1.f,          -1 * 1.f,
				1 * 1.f,           2 * 1.f,           4 * 1.f,           8 * 1.f,          16 * 1.f,          32 * 1.f,
				1 * 1.f,          -2 * 1.f,           4 * 1.f,          -8 * 1.f,          16 * 1.f,         -32 * 1.f,
				1 * 1.f,		   0.5*1.f,			 0.25*1.f,		   0.125*1.f,		 0.0625*1.f,	   0.03125*1.f,
				1 * 1.f,		   -0.5*1.f,         0.25*1.f,			-0.125*1.f,      0.0625*1.f,	  -0.03125*1.f,
				0 * 1.f,           0 * 1.f,           0 * 1.f,           0 * 1.f,           0 * 1.f,           1 * 1.f,
			};
			return A;
		}

		static const float *getB() {
			static const float B[T*T] = {
				1 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,
				0 * 1.f,         1 * 1.f,        -1 * 1.f,       0.5*1.f,      -0.5*1.f,         2 * 1.f,        -2 * 1.f,        -1 * 1.f,
				-5.25*1.f,         1 * 1.f,         1 * 1.f,      0.25*1.f,      0.25*1.f,         4 * 1.f,         4 * 1.f,         0 * 1.f,
				0 * 1.f,     -4.25*1.f,      4.25*1.f,      -2.5*1.f,       2.5*1.f,      -2.5*1.f,       2.5*1.f,      5.25*1.f,
				5.25*1.f,     -4.25*1.f,     -4.25*1.f,     -1.25*1.f,     -1.25*1.f,        -5 * 1.f,        -5 * 1.f,         0 * 1.f,
				0 * 1.f,         1 * 1.f,        -1 * 1.f,         2 * 1.f,        -2 * 1.f,       0.5*1.f,      -0.5*1.f,     -5.25*1.f,
				-1 * 1.f,         1 * 1.f,         1 * 1.f,         1 * 1.f,         1 * 1.f,         1 * 1.f,         1 * 1.f,         0 * 1.f,
				0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         0 * 1.f,         1 * 1.f,
			};
			return B;
		};
	};

	template<>
	struct WinogradParameters<WT_8X8_F_4X4_5X5>
	{

	private:
		// wt8x8, F(4x4,5x5)
		static const int T = 5 + 4 - 1;
		static const int K = 5;
		static const int O = 4;

	public:
		static const float *get(WINOGRAD_MATRIX mat, int &row, int& col) {

#if DEBUG_WINOGRAD
			assert(mat >= WINOGRAD_A && mat <= WINOGRAD_G);
#endif
			switch (mat) {

			case WINOGRAD_A: row = T; col = O; return getA();
			case WINOGRAD_B: row = T; col = T; return getB();
			case WINOGRAD_G: row = T; col = K; return getG();

			}

		}

	private:

		// from https://github.com/Maratyszcza/NNPACK/issues/12

		static const float *getG() {
			static const float G[T*K] = {
				1,       0,       0,       0,        0,
				-2. / 9., -2. / 9., -2. / 9., -2. / 9., -2. / 9.,
				-2. / 9.,  2. / 9., -2. / 9.,  2. / 9., -2. / 9.,
				1. / 90.,  1. / 45.,  2. / 45.,  4. / 45.,  8. / 45.,
				1. / 90., -1. / 45.,  2. / 45., -4. / 45.,  8. / 45.,
				4. / 45.,  2. / 45.,  1. / 45.,  1. / 90.,  1. / 180.,
				4. / 45., -2. / 45.,  1. / 45., -1. / 90.,  1. / 180.,
				0,       0,       0,       0,        1,
			};
			return G;
		}




		static const float *getA() {
			static const float A[T*(O)] = {
				1,  0, 0,  0,
				1,  1, 1,  1,
				1, -1, 1, -1,
				1,  2, 4,  8,
				1, -2, 4, -8,
				8,  4, 2,  1,
				8, -4, 2, -1,
				0,  0, 0,  1
			};
			return A;
		}

		static const float *getB() {
			static const float B[T*T] = {
				1,      0,      0,     0,     0,     0,     0,      0,
				0,      1,     -1,  1. / 2, -1. / 2,     2,    -2,     -1,
				-21. / 4,      1,      1,  1. / 4,  1. / 4,     4,     4,      0,
				0, -17. / 4,  17. / 4, -5. / 2,  5. / 2, -5. / 2,  5. / 2,  21. / 4,
				21. / 4, -17. / 4, -17. / 4, -5. / 4, -5. / 4,    -5,    -5,      0,
				0,      1,     -1,     2,    -2,  1. / 2, -1. / 2, -21. / 4,
				-1,      1,      1,     1,     1,     1,     1,      0,
				0,      0,      0,     0,     0,     0,     0,      1,
			};
			return B;
		}
	};

	class Winograd_Kron
	{

	private:

		Winograd_Kron(WINOGRAD_ALG alg, WINOGRAD_MATRIX mat) {

			isCalc = false;

			switch (alg) {

			case WT_8X8_F_6X6_3X3:
				matrix = WinogradParameters<WT_8X8_F_6X6_3X3>::get(mat, row, col); break;
			case WT_6X6_F_4X4_3X3:
				matrix = WinogradParameters<WT_6X6_F_4X4_3X3>::get(mat, row, col); break;
			case WT_8X8_F_4X4_5X5:
				matrix = WinogradParameters<WT_8X8_F_4X4_5X5>::get(mat, row, col); break;

			}

		}

	private:
		const float *matrix; // = A, B, G
		int row, col;// matrix: row*col
						// A: T*O
						// B: M*M
						// G: T*K

		boost::shared_ptr<caffe::Blob<float> > kron;

		bool isCalc;

	public:

		static Winograd_Kron *getInstance(WINOGRAD_ALG alg, WINOGRAD_MATRIX mat) {

			// 9 instances 3*3
			static Winograd_Kron * instances[WINOGRAD_ALG_NUM *WINOGRAD_MATRIX_NUM] = { NULL }; // according to [WINOGRAD_MATRIX] [WINOGRAD_PAIR]
	
			int index = alg*WINOGRAD_MATRIX_NUM + mat;

			if (instances[index] == NULL)
				instances[index] = new Winograd_Kron(alg, mat);

			return instances[index];

		}

		const boost::shared_ptr<caffe::Blob<float> > get() {
			if (isCalc)
				return kron;
			else {
				calc();
				return kron;
			}

		}

	private:

		void calc() {

			kron = boost::shared_ptr<caffe::Blob<float> >(new caffe::Blob<float>(shape));

			kronecker_product(kron->mutable_cpu_data(), matrix, matrix, row, col, row, col);

			isCalc = true;

		}

	};

	void kronecker_product(float *out, const float *in1, const float *in2, int m, int n, int p, int q)
	{
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int k = 0; k < p; ++k) {
					for (int l = 0; l < q; ++l) {
						out[(p*i + k)*n*q + q*j + l] = in1[n*i + j] * in2[k*q + l];
						/* compute in float precision in inference */
					}
				}
			}
		}
	}

	void winograd2D_initialize() {
		//singleton, precomputation before inference  

		Winograd_Kron::getInstance(WT_6X6_F_4X4_3X3, WINOGRAD_A)->get();
		Winograd_Kron::getInstance(WT_6X6_F_4X4_3X3, WINOGRAD_B)->get();
		Winograd_Kron::getInstance(WT_6X6_F_4X4_3X3, WINOGRAD_G)->get();

		Winograd_Kron::getInstance(WT_8X8_F_6X6_3X3, WINOGRAD_A)->get();
		Winograd_Kron::getInstance(WT_8X8_F_6X6_3X3, WINOGRAD_B)->get();
		Winograd_Kron::getInstance(WT_8X8_F_6X6_3X3, WINOGRAD_G)->get();
	}
}


#endif