
// Multiply two matrices A * B = C
 
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <oclUtils.h>
 
#define WA 642//1080
#define HA 480//1080//9
#define WB 642//1080//3
#define HB WA
#define WC WB
#define HC HA

extern "C"
void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB);

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

void profileEvent(cl_event &event)
{
    double dSeconds = executionTime(event);
    double dSize = ((double)WA * (double)HA * (double)WB * (double)HB);
    shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, %s Throughput = %.4f, Time = %.5f, Size = %.0f\n", 
            __func__, 1.0e-9 * dSize/dSeconds, dSeconds, dSize);
}

void profileTime()
{
    double dSeconds = shrDeltaT(0);
    double dSize = ((double)WA * (double)HA * (double)WB * (double)HB);
    shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, %s Throughput = %.4f, Time = %.5f, Size = %.0f\n", 
            __func__, 1.0e-9 * dSize/dSeconds, dSeconds, dSize);
}

void printDiff(float *data1, float *data2, int width, int height)
{
    int i,j,k;
    int error_count=0;
    for (j=0; j<height; j++) {
        for (i=0; i<width; i++) {
            k = j*width+i;
            if ( fabs(data1[k] - data2[k]) > 1e-5) {
                shrLog(LOGBOTH, 0, "diff(%d,%d) CPU=%f, GPU=%f \n", i,j, data1[k], data2[k]);
                error_count++;
            }
        }
    }
    shrLog(LOGBOTH, 0, " \nTotal Errors = %d \n", error_count);
}
/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int
main(int argc, char** argv)
{

    // set seed for rand()
    srand(time(NULL));

    // 1. allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // 2. initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

#if 0
    // 3. print out A and B
    printf("\n\nMatrix A\n");
    for(int i = 0; i < size_A; i++)
    {
      printf("%f ", h_A[i]);
      if(((i + 1) % WA) == 0)
      printf("\n");
    }

    printf("\n\nMatrix B\n");
    for(int i = 0; i < size_B; i++)
    {
      printf("%f ", h_B[i]);
      if(((i + 1) % WB) == 0)
      printf("\n");
    }
#endif
    // 4. allocate host memory for the result C
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // 5. Initialize OpenCL
    // OpenCL specific variables
    cl_context clGPUContext;
    cl_command_queue clCommandQue;
    cl_program clProgram;
    cl_kernel clKernel;

    size_t dataBytes;
    size_t kernelLength;
    cl_int errcode;

    // OpenCL device memory for matrices
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_C;

    /*****************************************/
    /* Initialize OpenCL */
    /*****************************************/
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    clGPUContext = clCreateContextFromType(0, 
                   CL_DEVICE_TYPE_GPU, 
                   NULL, NULL, &errcode);
    shrCheckError(errcode, CL_SUCCESS);

    // get the list of GPU devices associated 
    // with context
    errcode = clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, 0, NULL, 
              &dataBytes);
    cl_device_id *clDevices = (cl_device_id *)
              malloc(dataBytes);
    errcode |= clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, dataBytes, 
              clDevices, NULL);
    shrCheckError(errcode, CL_SUCCESS);

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, 
                  clDevices[0], CL_QUEUE_PROFILING_ENABLE, &errcode);
    shrCheckError(errcode, CL_SUCCESS);

    // Setup device memory
    d_C = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE, 
          mem_size_C, NULL, &errcode);
    d_A = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
          mem_size_A, h_A, &errcode);
    d_B = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
          mem_size_B, h_B, &errcode);


    // 6. Load and build OpenCL kernel
    char *clMatrixMul = oclLoadProgSource("matrixMul.cl",
                        "// My comment\n", 
                        &kernelLength);
    shrCheckError(clMatrixMul != NULL, shrTRUE);

    clProgram = clCreateProgramWithSource(clGPUContext, 
                1, (const char **)&clMatrixMul, 
                &kernelLength, &errcode);
    shrCheckError(errcode, CL_SUCCESS);

    errcode = clBuildProgram(clProgram, 0, 
              NULL, NULL, NULL, NULL);
    shrCheckError(errcode, CL_SUCCESS);

    clKernel = clCreateKernel(clProgram, 
               "matrixMul", &errcode);
    shrCheckError(errcode, CL_SUCCESS);


    // 7. Launch OpenCL kernel
    size_t localWorkSize[2], globalWorkSize[2];

    int wA = WA;
    int wC = WC;
    int wB = WB;
    int hA = HA;
    errcode = clSetKernelArg(clKernel, 0, 
              sizeof(cl_mem), (void *)&d_C);
    errcode |= clSetKernelArg(clKernel, 1, 
              sizeof(cl_mem), (void *)&d_A);
    errcode |= clSetKernelArg(clKernel, 2, 
              sizeof(cl_mem), (void *)&d_B);
    errcode |= clSetKernelArg(clKernel, 3, 
              sizeof(int), (void *)&hA);
    errcode |= clSetKernelArg(clKernel, 4, 
              sizeof(int), (void *)&wA);
    errcode |= clSetKernelArg(clKernel, 5, 
              sizeof(int), (void *)&wB);
/*    errcode |= clSetKernelArg(clKernel, 5, 
              sizeof(int), (void *)&wC);*/
    shrCheckError(errcode, CL_SUCCESS);

    localWorkSize[0] = 6;
    localWorkSize[1] = 6;
    globalWorkSize[0] = WB;  //wB
    globalWorkSize[1] = HA;  //hA

    cl_event GPUExecution;

    errcode = clEnqueueNDRangeKernel(clCommandQue, 
              clKernel, 2, NULL, globalWorkSize, 
              localWorkSize, 0, NULL, NULL);
    shrCheckError(errcode, CL_SUCCESS);

    // 8. Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQue, 
              d_C, CL_TRUE, 0, mem_size_C, 
              h_C, 0, NULL, &GPUExecution);
    shrCheckError(errcode, CL_SUCCESS);

    //clWaitForEvents(0, &GPUExecution);

    profileEvent(GPUExecution);
#if 0
    // 9. print out the results
    printf("\n\nMatrix C (Results)\n");
    for(int i = 0; i < size_C; i++)
    {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
    }
#endif
    // 10. allocate host memory for the golden C
    unsigned int size_gC = WC * HC;
    unsigned int mem_size_gC = sizeof(float) * size_gC;
    float* g_C = (float*) malloc(mem_size_gC);

    shrDeltaT(0);
    computeGold(g_C, h_A, h_B, HA, WA, WB);
    profileTime();
#if 0
    // 11. print out the results
    printf("\n\nGolden Matrix C\n");
    for(int i = 0; i < size_C; i++)
    {
      printf("%f ", g_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
    }
    printf("\n");
#endif
    // 11. compare results
    shrBOOL res = shrCompareL2fe(g_C, h_C, size_C, 1e-6f);
    shrLog(LOGBOTH, 0, "TEST %s \n\n", (1 == res) ? "PASSED" : "FAILED !!!");
    if (res != 1) 
    {
        printDiff(g_C, h_C, WC, HC);
    }

    // 11. clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(g_C);

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_B);

    free(clDevices);
    free(clMatrixMul);
    clReleaseContext(clGPUContext);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clCommandQue);

}


