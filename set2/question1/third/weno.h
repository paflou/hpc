#pragma once
#include <x86intrin.h>

void weno_minus_reference(const float * const a, const float * const b, const float * const c,
			  const float * const d, const float * const e, float * const out,
			  const int NENTRIES)
{
    for (int i=0; i<NENTRIES ; i+=8){
		__m256 a_ = _mm256_load_ps(a+i);
		__m256 b_ = _mm256_load_ps(b+i);
		__m256 c_ = _mm256_load_ps(c+i);
		__m256 d_ = _mm256_load_ps(d+i);
		__m256 e_ = _mm256_load_ps(e+i);

		__m256 coef1 = _mm256_set1_ps(4.0f/3.0f);
		__m256 coef2 = _mm256_set1_ps(19.0f/3.0f);
		__m256 coef3 = _mm256_set1_ps(11.0f/3.0f);
		__m256 coef4 = _mm256_set1_ps(25.0f/3.0f);
		__m256 coef5 = _mm256_set1_ps(31.0f/3.0f);
		__m256 coef6 = _mm256_set1_ps(10.0f/3.0f);
		__m256 coef7 = _mm256_set1_ps(13.0f/3.0f);
		__m256 coef8 = _mm256_set1_ps(5.0f/3.0f);

		// CALCULATION OF is0
		__m256 A = _mm256_mul_ps(a_, coef1);		//a*(4.0f/3.0f)
		A = _mm256_fmsub_ps(b_, coef2, A);			//a*(4.0f/3.0f) - b*(19.0f/3.0f)
		A = _mm256_fmadd_ps(c_, coef3, A);			//a*(4.0f/3.0f) - b*(19.0f/3.0f) + c*(11.0f/3.0f)
		A = _mm256_add_ps(a_, A);					//a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.))

		__m256 B = _mm256_mul_ps(b_, coef4);		//b*(25.0f/3.0f)
		B = _mm256_fmsub_ps(c_, coef5, B);			//b*(25.0f/3.0f) - c*(31.0f/3.0f)

		__m256 C = _mm256_mul_ps(c_, coef6);		//c*(10.0f/3.0f)
		C = _mm256_mul_ps(c_, C);					//c*c*(float)(10./3.)

		__m256 is0 = _mm256_add_ps(A, B);			//a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.))
		is0 = _mm256_add_ps(is0, C);				//a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.)


		// CALCULATION OF is1

		A = _mm256_mul_ps(b_, coef1);				//b*(4.0f/3.0f)
		A = _mm256_fmsub_ps(c_, coef7, A);			//b*(4.0f/3.0f) - c*(13.0f/3.0f)
		A = _mm256_fmadd_ps(d_, coef8, A);			//b*(4.0f/3.0f) - c*(13.0f/3.0f) + d*(5.0f/3.0f)
		A = _mm256_mul_ps(b_, A);					//b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))

		B = _mm256_mul_ps(c_, coef7);				//c*(13.0f/3.0f)
		B = _mm256_fmsub_ps(d_, coef7, B);			//c*(13.0f/3.0f) - d*(13.0f/3.0f)

		C = _mm256_mul_ps(d_, coef1);				//d*(4.0f/3.0f)
		C = _mm256_mul_ps(d_, C);					//d*d*(float)(4./3.)

		__m256 is1 = _mm256_add_ps(A, B);			//b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.)) + c*(13.0f/3.0f) - d*(13.0f/3.0f)
		is1 = _mm256_add_ps(is1, C);				//b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.)


		// CALCULATION OF is2

		A = _mm256_mul_ps(c_, coef6);				//c*(10.0f/3.0f)
		A = _mm256_fmsub_ps(d_, coef5, A);			//c*(10.0f/3.0f) - d*(31.0f/3.0f)
		A = _mm256_fmadd_ps(e_, coef3, A);			//c*(10.0f/3.0f) - d*(31.0f/3.0f) + e*(11.0f/3.0f)
		A = _mm256_mul_ps(c_, A);					//c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.))

		B = _mm256_mul_ps(d_, coef4);				//d*(25.0f/3.0f)
		B = _mm256_fmsub_ps(e_, coef2, B);			//d*(25.0f/3.0f) - e*(19.0f/3.0f)

		C = _mm256_mul_ps(e_, coef1);				//e*(4.0f/3.0f)
		C = _mm256_mul_ps(e_, C);					//e*e*(float)(4./3.)

		__m256 is2 = _mm256_add_ps(A, B);			//c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.))
		is2 = _mm256_add_ps(is2, C);				//c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.)




		__m256 weno = _mm256_set1_ps(WENOEPS);

		__m256 is0plus = _mm256_add_ps(is0, weno);
		__m256 is1plus = _mm256_add_ps(is1, weno);
		__m256 is2plus = _mm256_add_ps(is2, weno);

		__m256 alpha0 = _mm256_mul_ps(_mm256_set1_ps(0.1), _mm256_rcp_ps(_mm256_mul_ps(is0plus, is0plus)));
		__m256 alpha1 = _mm256_mul_ps(_mm256_set1_ps(0.6), _mm256_rcp_ps(_mm256_mul_ps(is1plus, is1plus)));
		__m256 alpha2 = _mm256_mul_ps(_mm256_set1_ps(0.3), _mm256_rcp_ps(_mm256_mul_ps(is2plus, is2plus)));
		__m256 alphasum = _mm256_add_ps(alpha0, alpha1);
		alphasum = _mm256_add_ps(alphasum, alpha2);

		__m256 inv_alpha = _mm256_rcp_ps(alphasum);

		__m256 omega0 = _mm256_mul_ps(alpha0, inv_alpha);
		__m256 omega1 = _mm256_mul_ps(alpha1, inv_alpha);
		__m256 omega2 = _mm256_sub_ps(_mm256_set1_ps(1), omega0);
		omega2 = _mm256_sub_ps(omega2, omega1);


		coef1 = _mm256_set1_ps(1.0f/3.0f);
		coef2 = _mm256_set1_ps(7.0f/6.0f);
		coef3 = _mm256_set1_ps(11.0f/6.0f);
		coef4 = _mm256_set1_ps(1.0f/6.0f);
		coef5 = _mm256_set1_ps(5.0f/6.0f);

		A = _mm256_mul_ps(coef1, a_);			//(float)(1./3.)*a
		A = _mm256_fmsub_ps(coef2, b_, A);		//(float)(1./3.)*a - (float)(7./6.)*b
		A = _mm256_fmadd_ps(coef3,c_,A);		//(float)(1./3.)*a - (float)(7./6.)*b + (float)(11./6.)*c
		A = _mm256_mul_ps(A,omega0);			//omega0*((float)(1./3.)*a - (float)(7./6.)*b + (float)(11./6.)*c)

		B = _mm256_mul_ps(coef5,c_);			//(float)(5./6.)*c
		B = _mm256_fmsub_ps(coef4, b_, B);		//-(float)(1./6.)*b + (float)(5./6.)*c
		B = _mm256_fmadd_ps(coef1, d_, B);		//-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d
		B = _mm256_mul_ps(B, omega1);			//omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d)

		C = _mm256_mul_ps(coef1,c_);			//(float)(1./3.)*c
		C = _mm256_fmsub_ps(coef4, e_, C);		//(float)(1./3.)*c - (float)(1./6.)*e
		C = _mm256_fmadd_ps(coef5, d_, C);		//(float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e
		C = _mm256_mul_ps(C, omega2);			//omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e)

		__m256 res = _mm256_add_ps(A, B);
		res = _mm256_add_ps(res, C);

		
		_mm256_store_ps(&out[i], res);
	}
}
