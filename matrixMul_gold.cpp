/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <CL/opencl.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {

			const unsigned ix = j;
			const unsigned iy = i;
			const buffer_w = wB;

			const unsigned N = 4;
			const unsigned ix_rounded_to_prev_N_multiple = ix / N * N;
			const unsigned iy_rounded_to_prev_N_multiple = iy / N * N;

			const unsigned ia = ix_rounded_to_prev_N_multiple + buffer_w * iy;
			const unsigned ib = ix_rounded_to_prev_N_multiple + buffer_w * (iy_rounded_to_prev_N_multiple + ix - ix_rounded_to_prev_N_multiple);

            const float a = A[ia];
            const float b = B[ib];
            float sum = a * b;

            for (unsigned k = 1; k < N; ++k) {
                const float a = A[ia + k];
                const float b = B[ib + k];
                sum += a * b;
            }
            C[i * buffer_w + j] = sum;
        }
}
