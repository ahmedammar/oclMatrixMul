#define buffer_w	16
#define buffer_h	16
#define N			4

// OpenCL Kernel
__kernel void
matrixMul(__global float* C, 
          __global float* A, 
          __global float* B)
{
    // block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

	// work-group[]: { 2, 2 }
	int ix = bx * 2 + tx;
	int iy = by * 2 + ty;

	int ix_rounded_to_prev_N_multiple = ix / N * N;
	int iy_rounded_to_prev_N_multiple = iy / N * N;

	int ia = ix_rounded_to_prev_N_multiple + buffer_w * iy;
	int ib = ix_rounded_to_prev_N_multiple + buffer_w * (iy_rounded_to_prev_N_multiple + ix - ix_rounded_to_prev_N_multiple);

    // write one matrix element
    C[iy * buffer_w + ix] = dot(
		(float4)(
			A[ia + 0],
			A[ia + 1],
			A[ia + 2],
			A[ia + 3]),
		(float4)(
			B[ib + 0],
			B[ib + 1],
			B[ib + 2],
			B[ib + 3]));
}

