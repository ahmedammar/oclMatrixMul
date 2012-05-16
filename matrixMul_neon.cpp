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

    float32x4_t Arow[4], Bcol[4], C1row[4], C2row[4], C3row[4], C4row[4], tmp;
    float32x4x2_t trn1, trn2, Brow1, Brow2;
    uint32_t i, j, k;

    // width(A) = height(B)!
    // Loop over A's rows, process 4 rows in each iteration
    for (i = 0; i < hA; i+=4) {

        /* | a_11 a_12 a_13 a_14 ... a_1M |
         * | a_21 a_22 a_23 a_24 ... a_2M |
         * | a_31 a_32 a_33 a_34 ... a_3M |
         * | a_41 a_42 a_43 a_44 ... a_4M |
         * |  .               .  ...   .  |
         * | a_N1 a_N2 a_N3 a_N4 ... a_NM |
         *
         * First block to process is:
         * | a_11 a_12 a_13 a_14 |
         * | a_21 a_22 a_23 a_24 |
         * | a_31 a_32 a_33 a_34 |
         * | a_41 a_42 a_43 a_44 |
         *
         * We'll have to multiply that with B's first
         * 4x4 submatrix, only transposed. So we need:
         * | b_11 b_21 b_31 b_41 |
         * | b_12 b_22 b_32 b_42 |
         * | b_13 b_23 b_33 b_43 |
         * | b_14 b_24 b_34 b_44 |
         *
         * The result for C, will be the result of the following
         * operations:
         * | a_11*b_11 a_12*b_21 a_13*b_31 a_14*b_41 |
         * | a_21*b_12 a_22*b_22 a_23*b_32 a_24*b_42 |
         * | a_31*b_13 a_32*b_23 a_33*b_33 a_34*b_43 |
         * | a_41*b_14 a_42*b_24 a_43*b_34 a_44*b_44 |
         *
         * Now transpose this and add per row, the result is the
         * following 4 column vector
         * | a_11*b_11 + a_12*b_21 + a_13*b_31 + a_14*b_41 |
         * | a_21*b_12 + a_22*b_22 + a_23*b_32 + a_24*b_42 |
         * | a_31*b_13 + a_32*b_23 + a_33*b_33 + a_34*b_43 |
         * | a_41*b_14 + a_42*b_24 + a_43*b_34 + a_44*b_44 |
         *
         * Loop over the rest of A'columns/B's rows and repeat
         * the process and add to the previous vector to get
         * | c_11 c_12 c_13 c_14 |
         *
         */
         // loop over B's columns
        for (k = 0; k < wB; k+=4) {
            C1row[0] = vdupq_n_f32(0.0);
            C1row[1] = vdupq_n_f32(0.0);
            C1row[2] = vdupq_n_f32(0.0);
            C1row[3] = vdupq_n_f32(0.0);
            C2row[0] = vdupq_n_f32(0.0);
            C2row[1] = vdupq_n_f32(0.0);
            C2row[2] = vdupq_n_f32(0.0);
            C2row[3] = vdupq_n_f32(0.0);
            C3row[0] = vdupq_n_f32(0.0);
            C3row[1] = vdupq_n_f32(0.0);
            C3row[2] = vdupq_n_f32(0.0);
            C3row[3] = vdupq_n_f32(0.0);
            C4row[0] = vdupq_n_f32(0.0);
            C4row[1] = vdupq_n_f32(0.0);
            C4row[2] = vdupq_n_f32(0.0);
            C4row[3] = vdupq_n_f32(0.0);

            for (j = 0; j < wA; j+=4) {
                __pld(&A[i*wA + j]);
                Arow[0] = vld1q_f32(&A[i*wA + j]);
                Arow[1] = vld1q_f32(&A[(i+1)*wA + j]);
                Arow[2] = vld1q_f32(&A[(i+2)*wA + j]);
                Arow[3] = vld1q_f32(&A[(i+3)*wA + j]);
                Bcol[0] = vld1q_f32(&B[j*wB + k]);
                Bcol[1] = vld1q_f32(&B[(j+1)*wB + k]);
                Bcol[2] = vld1q_f32(&B[(j+2)*wB + k]);
                Bcol[3] = vld1q_f32(&B[(j+3)*wB + k]);

                // Transpose Bcol_v 4x4 submatrix
                trn1 = vzipq_f32(Bcol[0], Bcol[2]);
                trn2 = vzipq_f32(Bcol[1], Bcol[3]);
                Brow1 = vzipq_f32(trn1.val[0], trn2.val[0]);
                Brow2 = vzipq_f32(trn1.val[1], trn2.val[1]);

                // Do row |c_ij c_i(j+1) c_i(j+2) c_i(j+3) |
                C1row[0] = vmlaq_f32(C1row[0], Arow[0], Brow1.val[0]);
                C1row[1] = vmlaq_f32(C1row[1], Arow[0], Brow1.val[1]);
                C1row[2] = vmlaq_f32(C1row[2], Arow[0], Brow2.val[0]);
                C1row[3] = vmlaq_f32(C1row[3], Arow[0], Brow2.val[1]);

                // Do row |c_(i+1)j c_(i+1)(j+1) c_(i+1)(j+2) c_(i+1)(j+3) |
                C2row[0] = vmlaq_f32(C2row[0], Arow[1], Brow1.val[0]);
                C2row[1] = vmlaq_f32(C2row[1], Arow[1], Brow1.val[1]);
                C2row[2] = vmlaq_f32(C2row[2], Arow[1], Brow2.val[0]);
                C2row[3] = vmlaq_f32(C2row[3], Arow[1], Brow2.val[1]);

                // Do row |c_(i+2)j c_(i+2)(j+1) c_(i+2)(j+2) c_(i+2)(j+3) |
                C3row[0] = vmlaq_f32(C3row[0], Arow[2], Brow1.val[0]);
                C3row[1] = vmlaq_f32(C3row[1], Arow[2], Brow1.val[1]);
                C3row[2] = vmlaq_f32(C3row[2], Arow[2], Brow2.val[0]);
                C3row[3] = vmlaq_f32(C3row[3], Arow[2], Brow2.val[1]);

                // Do row |c_(i+3)j c_(i+3)(j+1) c_(i+3)(j+2) c_(i+3)(j+3) |
                C4row[0] = vmlaq_f32(C4row[0], Arow[3], Brow1.val[0]);
                C4row[1] = vmlaq_f32(C4row[1], Arow[3], Brow1.val[1]);
                C4row[2] = vmlaq_f32(C4row[2], Arow[3], Brow2.val[0]);
                C4row[3] = vmlaq_f32(C4row[3], Arow[3], Brow2.val[1]);
            }

            trn1 = vzipq_f32(C1row[0], C1row[2]);
            trn2 = vzipq_f32(C1row[1], C1row[3]);
            Brow1 = vzipq_f32(trn1.val[0], trn2.val[0]);
            Brow2 = vzipq_f32(trn1.val[1], trn2.val[1]);

            C1row[0] = vaddq_f32(Brow1.val[0], Brow1.val[1]);
            tmp = vaddq_f32(C1row[0], Brow2.val[0]);
            C1row[0] = vaddq_f32(tmp, Brow2.val[1]);

            trn1 = vzipq_f32(C2row[0], C2row[2]);
            trn2 = vzipq_f32(C2row[1], C2row[3]);
            Brow1 = vzipq_f32(trn1.val[0], trn2.val[0]);
            Brow2 = vzipq_f32(trn1.val[1], trn2.val[1]);

            C1row[1] = vaddq_f32(Brow1.val[0], Brow1.val[1]);
            tmp = vaddq_f32(C1row[1], Brow2.val[0]);
            C1row[1] = vaddq_f32(tmp, Brow2.val[1]);

//          C1row[1] = vaddq_f32(vaddq_f32(Brow1.val[0], Brow1.val[1]), vaddq_f32(Brow2.val[0], Brow2.val[1]));

            trn1 = vzipq_f32(C3row[0], C3row[2]);
            trn2 = vzipq_f32(C3row[1], C3row[3]);
            Brow1 = vzipq_f32(trn1.val[0], trn2.val[0]);
            Brow2 = vzipq_f32(trn1.val[1], trn2.val[1]);

            C1row[2] = vaddq_f32(Brow1.val[0], Brow1.val[1]);
            tmp = vaddq_f32(C1row[2], Brow2.val[0]);
            C1row[2] = vaddq_f32(tmp, Brow2.val[1]);
//            C1row[2] = vaddq_f32(vaddq_f32(Brow1.val[0], Brow1.val[1]), vaddq_f32(Brow2.val[0], Brow2.val[1]));

            trn1 = vzipq_f32(C4row[0], C4row[2]);
            trn2 = vzipq_f32(C4row[1], C4row[3]);
            Brow1 = vzipq_f32(trn1.val[0], trn2.val[0]);
            Brow2 = vzipq_f32(trn1.val[1], trn2.val[1]);

            C1row[3] = vaddq_f32(Brow1.val[0], Brow1.val[1]);
            tmp = vaddq_f32(C1row[3], Brow2.val[0]);
            C1row[3] = vaddq_f32(tmp, Brow2.val[1]);
//            C1row[3] = vaddq_f32(vaddq_f32(Brow1.val[0], Brow1.val[1]), vaddq_f32(Brow2.val[0], Brow2.val[1]));

            vst1q_f32(&C[i*wB + k], C1row[0]);
            vst1q_f32(&C[(i+1)*wB + k], C1row[1]);
            vst1q_f32(&C[(i+2)*wB + k], C1row[2]);
            vst1q_f32(&C[(i+3)*wB + k], C1row[3]);
        }
    }
}
