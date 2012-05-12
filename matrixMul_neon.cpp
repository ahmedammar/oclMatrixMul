#include <sys/times.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <arm_neon.h>
#include <CL/opencl.h>

#ifndef __pld
#define __pld(x) asm volatile ( "   pld [%[addr]]\n" :: [addr] "r" (x) : "cc" );
#endif

int computeNeon(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB) {


	float32x4_t row_v, col_v, product_v;
	float32x2_t product_lo, product_hi, sum;
	float s[2];
	uint32_t i, j, k;

	for (i = 0; i < hA; i++) {
		for(j = 0; j< wB; j++) {
#if 0
			// First splat V[x] in V_v.
			col_v1 = vdup_n_u16(B[j*wB].s0);
			col_v2 = vdup_n_u16(B[j*wB].s4);
#endif
			
			// Initialize product_v vector to zero
			product_v = vdupq_n_f32(0.0f);
			for (k = 0; k < wA; k++) {
				// Load row's next 8 16bit ints into row_v
				row_v = vld1q_f32(&A[i*wA + k]);

				col_v = vld1q_f32(&B[k*wB + j]);

				// Multiply V[x] with 4 row[x] elements 
				// add to previous product_v and store back the result
				product_v = vmlaq_f32(product_v, row_v, col_v);
			}
			// Now add all elements and store the result to a sum vector
			// and then store that to s[2] array. The sum is in s[0]
			product_lo = vget_low_f32(product_v);
			product_hi = vget_high_f32(product_v);
			sum = vpadd_f32(product_lo, product_hi);
			sum = vpadd_f32(sum, sum);
			vst1_f32(s, sum);

			C[i * wB + j] = s[0] + s[1];
		}
	}

	return 0;
}
