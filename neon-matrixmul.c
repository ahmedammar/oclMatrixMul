#include <sys/times.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

struct timespec diff(struct timespec start, struct timespec end)
{
        struct timespec temp;
        if ((end.tv_nsec-start.tv_nsec)<0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = end.tv_nsec-start.tv_nsec;
        } else {
                temp.tv_sec = end.tv_sec-start.tv_sec;
                temp.tv_nsec = end.tv_nsec-start.tv_nsec;
        }
        return temp;
}

#include <arm_neon.h>

#ifndef __pld
#define __pld(x) asm volatile ( "   pld [%[addr]]\n" :: [addr] "r" (x) : "cc" );
#endif

#define LOOPS   10
#define WIDTH	256
#define HEIGHT	256

#define WIDTHF  256.0
#define HEIGHTF 256.0

float __attribute__((aligned(16))) matrix[WIDTH*HEIGHT];
float __attribute__((aligned(16))) A[WIDTH*HEIGHT];
float __attribute__((aligned(16))) B[WIDTH*HEIGHT];
float __attribute__((aligned(16))) C1[WIDTH*HEIGHT];
float __attribute__((aligned(16))) C2[WIDTH*HEIGHT];
float __attribute__((aligned(16))) C3[WIDTH*HEIGHT];
float __attribute__((aligned(16))) vector[WIDTH];
float __attribute__((aligned(16))) result[HEIGHT];

void printMatrix(char *name, const float* A, size_t hA, size_t wA, size_t rwA) {
    size_t i, j;
    printf("\n%s:\n", name);
    for (i = 0; i < hA; ++i) {
        for (j = 0; j < wA; ++j) {
            printf(" %f", A[i*rwA + j]);
        }
        printf("\n");
    }
}

int32_t intDiff(float a, float b) {
    int32_t intDiff = abs(*(int32_t*)&a - *(int32_t*)&b);
    return intDiff;
}

void diffMatrix(const float* A, const float* B, size_t h, size_t w) {
    size_t i, j;
    float absdiff = 0.0, diff;
    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            if (A[i*w +  j] != B[i*w + j]) {
//                printf("intDiff = %d\n", intDiff(A[i*w + j], B[i*w + j]));
                diff = fabs(A[i*w + j] - B[i*w + j]);
//                printf("A[%d][%d] = %f != B[%d][%d] = %f, diff = %f\n", i, j, A[i*w + j], i, j, B[i*w + j], diff);
                absdiff += diff;
            }
        }
    }
    printf("absdiff = %f\n", absdiff);
}

int MatrixVectorMul(const float* M, uint32_t width, uint32_t height,
                              const float* V, float* W) {

	float32x4_t row_v, col_v, product_v;
	float32x2_t product_lo, product_hi, sum;
	float s[2];
	uint32_t x, y;

	// Each work-group computes multiple elements of W
	for (y = 0; y < height; y++) {
		// First splat V[x] in V_v.
		col_v = vdupq_n_f32(V[y]);

		// Initialize product_v vector to zero
		product_v = vdupq_n_f32(0.0f);
		for (x = 0; x < width; x += 4) {
			// Load row's next 4 floats into row_v
			__pld(&M[y*width + x]);
			row_v = vld1q_f32(&M[y*width + x]);

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

		W[y] = s[0];
	}
}

void MatrixMatrixMul(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    size_t i, j, k;
    for (i = 0; i < hA; ++i)
        for (j = 0; j < wB; ++j) {
            float sum = 0.0f;
            for (k = 0; k < wA; ++k) {
                float a = A[i * wA + k];
                float b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = sum;
        }
}

int MatrixMatrixMul_neon(float* C, const float* A, const float* B, uint32_t hA, uint32_t wA, uint32_t wB) {

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

int MatrixMatrixMul_neon2(float* C, const float* A, const float* B, uint32_t hA, uint32_t wA, uint32_t wB) {

	float32x4_t row_v, col_v[4], product_v;
	float32x2_t product_lo, product_hi, sum;
	float s[2];
	uint32_t i, j, k;

	for (i = 0; i < hA; i++) {
		for(j = 0; j< wB; j+=4) {
			row_v = vdupq_n_f32(B[j*wA + k]);
			// Initialize product_v vector to zero
			product_v = vdupq_n_f32(0.0f);
			for (k = 0; k < wA; k++) {
				// Load next row from A matrix
				row_v = vld1q_f32(&A[i*wA + k]);

				// Multiply V[x] with 4 row[x] elements 
				// add to previous product_v and store back the result
				product_v = vmlaq_f32(product_v, row_v, col_v[4]);
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

void benchmark_function(double size, void (*function)()) {
        double bw, fps;
        struct timespec t00, t11, res;

        clock_gettime(CLOCK_MONOTONIC, &t00);
        (*function)();
        clock_gettime(CLOCK_MONOTONIC, &t11);
        uint32_t dts = diff(t00,t11).tv_sec;
        double dtn = (double) diff(t00,t11).tv_nsec / 1000000000;
        printf("dts: %i\n", dts);
        printf("dtn: %lf\n", dtn);
        dtn += ((double)dts);
        printf("dts: %lf\n", dtn);
        bw = size / (1048576.0 * dtn);
        fps = 1.0 / dtn;

        printf("dt: %lf secs, bw: %lf MFLOPS, size = %lf\n", dtn, bw, size);

        printf("Ability to process Approx. %lf frames per sec\n", fps);
}

void do_matrixvectormul() {
    MatrixVectorMul(matrix, WIDTH, HEIGHT, vector, result);
}

void do_matrixmatrixmul() {
    int i;
    for (i=0; i < LOOPS; i++)
        MatrixMatrixMul(C1, A, B, HEIGHT, WIDTH, HEIGHT);
}

void do_matrixmatrixmul_neon() {
    int i;
    for (i=0; i < LOOPS; i++)
        MatrixMatrixMul_neon(C2, A, B, HEIGHT, WIDTH, HEIGHT);
}

void do_matrixmatrixmul_neon2() {
    int i;
    for (i=0; i < LOOPS; i++)
        MatrixMatrixMul_neon2(C3, A, B, HEIGHT, WIDTH, HEIGHT);
}

void init_array(float *M, size_t size) {
	int i;
	for (i=0; i < size; i++) {
		M[i] = 0.5*((float)rand()/(float)INT_MAX);
        }
}

int main() {
	init_array(matrix, WIDTH*HEIGHT);
	init_array(A, WIDTH*HEIGHT);
	init_array(B, WIDTH*HEIGHT);
	init_array(vector, WIDTH);

//	printMatrix("A", A, HEIGHT, WIDTH, WIDTH);
//	printMatrix("B", B, HEIGHT, WIDTH, WIDTH);
//	benchmark_function(WIDTHF*HEIGHTF, do_matrixvectormul);
        printf("Scalar:\n");
	benchmark_function(WIDTHF*HEIGHTF*WIDTHF*HEIGHTF, do_matrixmatrixmul);
        printf("NEON:\n");
	benchmark_function(WIDTHF*HEIGHTF*WIDTHF*HEIGHTF, do_matrixmatrixmul_neon);
        printf("NEON2\n");
	benchmark_function(WIDTHF*HEIGHTF*WIDTHF*HEIGHTF, do_matrixmatrixmul_neon2);
        diffMatrix(C1, C2, WIDTH, HEIGHT);
        diffMatrix(C1, C3, WIDTH, HEIGHT);
}
	
