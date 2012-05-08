__kernel void
matrixMul(__global short* C, 
          __global short8* A, 
          __global short8* B)
{
    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local short8 As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the local memory array Bs 
    // used to store the sub-matrix of B
    __local short8 Bs[BLOCK_SIZE][BLOCK_SIZE];
 
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
    int bStep  = BLOCK_SIZE * WB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    short8 Csub = { 0, 0, 0, 0, 0, 0, 0, 0};//0.000001041666667 * WA; // We have some mathematical error?
 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    {

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + WA * ty + tx];
        Bs[ty][tx] = B[b + WB * ty + tx];
 
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    //int c = WB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub.s0 +
    Csub.s1 + Csub.s2 + Csub.s3 + Csub.s4 + Csub.s5 + Csub.s6 + Csub.s7;

}

