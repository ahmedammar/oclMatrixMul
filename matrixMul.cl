#if 0
__kernel void
matrixMulFULL(__global float* C, 
          __global float* A, 
          __global float* B)
{
    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the local memory array BS 
    // used to store the sub-matrix of B
    __local float BS[BLOCK_SIZE][BLOCK_SIZE];
 
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
 
    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = WA * BLOCK_SIZE * by;
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + WA - 1;
 
    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int BStep  = BLOCK_SIZE * WB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f; //0.000001041666667 * WA; // We have some mathematical error?
 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += BStep) 
    {

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + WA * ty + tx];
        BS[ty][tx] = B[b + WB * ty + tx];
 
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * BS[k][tx];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    //int c = WB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
}
#endif
#define WA 16
#define WB 16

#define N 4

// OpenCL Kernel
__kernel void
matrixMul(__global float* C, 
          __global float* A, 
          __global float* B)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // value stores the element 
    // that is computed by the thread
    float value = 0; //FIXME
    for (int k = 0; k < WA; ++k)
    {
      float elementA = A[(by*WA*3) + (ty*WA) + k];
      //float elementB = B[(bx*WB*3) + (tx*WB) + k]; //coalesced access to B
      float elementB = B[(bx*3) + tx + (WB*k)];
      value += elementA * elementB;
      //value = mad(elementA, elementB, value);
    }

    // Write the matrix to device memory each 
    // thread writes one element
    C[(WB*by*3) + (WB*ty) + (bx*3) + tx] = value;
}

