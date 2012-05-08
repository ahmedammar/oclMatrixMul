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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication with multi GPU support.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// standard utilities and system includes
#include <oclUtils.h>

// project include
#include "matrixMul.h"

#define GPU_PROFILING

// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 8;

// global variables
cl_context cxGPUContext;
cl_kernel multiplicationKernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, const char** argv);
void printDiff(cl_short*, cl_short*, int, int);
void print(cl_short8 *data1, int width, int height);
void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, cl_short8* h_B_data, unsigned int mem_size_B, cl_short* h_C );

extern "C"
void computeGold(cl_short*, const cl_short8*, const cl_short8*, unsigned int, unsigned int, unsigned int);
void computeNeon(cl_short*, const cl_short8*, const cl_short8*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv)
{
    // start the logs
    shrSetLogFileName ("oclMatrixMul.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    // run the code
    if (runTest(argc, argv) != 0)
    {
        shrLog(LOGBOTH, 0, "TEST FAILED !!!\n\n");
    }

    // finish
    shrEXIT(argc, argv);
}

void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, cl_short8* h_B_data, unsigned int mem_size_B, cl_short* h_C )
{
    cl_mem d_A[MAX_GPU_COUNT];
    cl_mem d_C[MAX_GPU_COUNT];
    cl_mem d_B[MAX_GPU_COUNT];

    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];

    // Start the computation on each available GPU
    
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    int sizePerGPU = HA / ciDeviceCount;

    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];

    workOffset[0] = 0;
    for(unsigned int i=0; i < ciDeviceCount; ++i) 
    {
        // Input buffer
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (HA - workOffset[i]);        

        d_A[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(cl_short8) * WA, NULL,NULL);

        // Copy only assigned rows from host to device
        clEnqueueCopyBuffer(commandQueue[i], h_A, d_A[i], workOffset[i] * sizeof(cl_short8) * WA, 
                            0, workSize[i] * sizeof(cl_short8) * WA, 0, NULL, NULL);        
        
        // create OpenCL buffer on device that will be initiatlize from the host memory on first use
        // on device
        d_B[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                mem_size_B, h_B_data, NULL);

        // Output buffer
        d_C[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  workSize[i] * WC * sizeof(cl_short8), NULL,NULL);
              
        // set the args values
        clSetKernelArg(multiplicationKernel[i], 0, sizeof(cl_mem), (void *) &d_C[i]);
        clSetKernelArg(multiplicationKernel[i], 1, sizeof(cl_mem), (void *) &d_A[i]);
        clSetKernelArg(multiplicationKernel[i], 2, sizeof(cl_mem), (void *) &d_B[i]);
        //clSetKernelArg(multiplicationKernel[i], 3, sizeof(cl_short8) * BLOCK_SIZE *BLOCK_SIZE, 0 );
        //clSetKernelArg(multiplicationKernel[i], 4, sizeof(cl_short8) * BLOCK_SIZE *BLOCK_SIZE, 0 );

        if(i+1 < ciDeviceCount)
            workOffset[i + 1] = workOffset[i] + workSize[i];
    }
    
    // Execute Multiplication on all GPUs in parallel
    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, WC), shrRoundUp(BLOCK_SIZE, workSize[0])};
//while(1) 
{
    // Start timer and launch kernels on devices
    shrDeltaT(0);
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        // Multiplication - non-blocking execution
        globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, workSize[i]);

	
        shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, localWorkSize: %d,%d globalWorkSize: %d,%d\n", localWorkSize[0], localWorkSize[1], globalWorkSize[0], globalWorkSize[1]);
        int ret = clEnqueueNDRangeKernel(commandQueue[i], multiplicationKernel[i], 2, 0, globalWorkSize, localWorkSize,
                               0, NULL, &GPUExecution[i]);
        shrLog(LOGBOTH, 0, "ret: %i\n", ret);
    }

    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        // Non-blocking copy of result from device to host
        clEnqueueReadBuffer(commandQueue[i], d_C[i], CL_FALSE, 0, WC * sizeof(cl_short8) * workSize[i], 
                            h_C + workOffset[i] * WC, 0, NULL, &GPUDone[i]);
    }

    // CPU sync with GPU
    clWaitForEvents(ciDeviceCount, GPUDone);
    //clFinish(commandQueue[0]);

    // stop and log timer 
    #ifdef GPU_PROFILING
        double dSeconds = shrDeltaT(0);
        double dSize = ((double)WA * (double)HA * (double)WB * (double)HB * 8 * 8);
        shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, [GPU] Throughput = %.4f, Time = %.5f, Size = %.0f, NumDevsUsed = %d, Workgroup = %u\n", 
                1.0e-9 * dSize/dSeconds, dSeconds, dSize, ciDeviceCount, localWorkSize[0] * localWorkSize[1]);

        // Print kernel timing per GPU
        for(unsigned int i = 0; i < ciDeviceCount; i++) 
        {    
            shrLog(LOGBOTH, 0, "  Kernel execution time on GPU %d \t: %.5f s\n", i, executionTime(GPUExecution[i]));
        }
        shrLog(LOGBOTH, 0, "\n");
     #endif
}
    // Release mem and event objects    
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        clReleaseMemObject(d_A[i]);
        clReleaseMemObject(d_C[i]);
        clReleaseMemObject(d_B[i]);
	    clReleaseEvent(GPUExecution[i]);
	    clReleaseEvent(GPUDone[i]);
    }
}

shrBOOL Compare(cl_short* ref, cl_short* data, int width, int height)
{
    int i,j,k;
    for (j=0; j<height; j++) {
        for (i=0; i<width; i++) {
          k = j*width+i;
        /*if (ref[i].s0 != data[i].s0) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s0, data[i].s0);
            return shrFALSE;
        }
        if (ref[i].s1 != data[i].s1) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s1, data[i].s1);
            return shrFALSE;
        }
        if (ref[i].s2 != data[i].s2) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s2, data[i].s2);
            return shrFALSE;
        }
        if (ref[i].s3 != data[i].s3) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s3, data[i].s3);
            return shrFALSE;
        }
        if (ref[i].s4 != data[i].s4) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s4, data[i].s4);
            return shrFALSE;
        }
        if (ref[i].s5 != data[i].s5) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s5, data[i].s5);
            return shrFALSE;
        }
        if (ref[i].s6 != data[i].s6) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s6, data[i].s6);
            return shrFALSE;
        }
        if (ref[i].s7 != data[i].s7) {
            shrLog(LOGBOTH, 0, "diff(%d) %i %i\n", i, ref[i].s7, data[i].s7);
            return shrFALSE;
        }*/
        if (ref[k] != data[k]) {
            shrLog(LOGBOTH, 0, "diff(%d,%d) %i %i\n", i, j, ref[k], data[k]);
            return shrFALSE;
        }
      }
    }
    return shrTRUE;
}

void FillArray(cl_short8* pfData, int iSize)
{
    int i;
    for (i = 0; i < iSize; ++i)
    {
        pfData[i].s0 = rand();
        pfData[i].s1 = rand();
        pfData[i].s2 = rand();
        pfData[i].s3 = rand();
        pfData[i].s4 = rand();
        pfData[i].s5 = rand();
        pfData[i].s6 = rand();
        pfData[i].s7 = rand();
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for 
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, const char** argv)
{
    cl_uint ciDeviceCount = 0;
    cl_int ciErrNum = CL_SUCCESS;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    // create the OpenCL context on available GPU devices
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to create OpenCL context!\n");
        return ciErrNum;
    }

    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        // User specified GPUs
        char* deviceList;
        char* deviceStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

        #ifdef WIN32
            deviceStr = strtok_s (deviceList," ,.-", &next_token);
        #else
            deviceStr = strtok (deviceList," ,.-");
        #endif   
        while(deviceStr != NULL) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
            shrLog(LOGBOTH, 0, "Device %d:\n", ciDeviceCount);
            oclPrintDevName(LOGBOTH, device);            
           
            // create command queue
            commandQueue[ciDeviceCount] = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(LOGBOTH, 0, " Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }

            #if 0 //GPU_PROFILING
                ciErrNum = clSetCommandQueueProperty(commandQueue[ciDeviceCount], CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
                if (ciErrNum != CL_SUCCESS)
                {
                    shrLog(LOGBOTH, 0, " Error %i in clSetCommandQueueProperty call !!!\n\n", ciErrNum);
                    return ciErrNum;
                }
            #endif
                
             ++ciDeviceCount;

            #ifdef WIN32
                deviceStr = strtok_s (NULL," ,.-", &next_token);
            #else            
                deviceStr = strtok (NULL," ,.-");
            #endif
        }

        free(deviceList);
    } 
    else 
    {
        // Find out how many GPU's to compute on all available GPUs
	    size_t nDeviceBytes;
	    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(LOGBOTH, 0, " Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            return ciErrNum;
        }
        else if (ciDeviceCount == 0)
        {
            shrLog(LOGBOTH, 0, " There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
            return -1;
        } 

        // create command-queues
        for(unsigned int i = 0; i < ciDeviceCount; ++i) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, i);
            shrLog(LOGBOTH, 0, "Device %d:\n", i);
            oclPrintDevName(LOGBOTH, device);            
            
            // create command queue
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(LOGBOTH, 0, " Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
            #if 0 //def GPU_PROFILING
                clSetCommandQueueProperty(commandQueue[i], CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
            #endif
        }
    }

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(cl_short8) * size_A;
    cl_short8* h_A_data = (cl_short8*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(cl_short8) * size_B;
    cl_short8* h_B_data = (cl_short8*) malloc(mem_size_B);

    // initialize host memory
    srand(time(NULL));//2006);
    FillArray(h_A_data, size_A);
    FillArray(h_B_data, size_B);

   //print(h_A_data, WA, HA);

   //print(h_B_data, WB, HB);
    // allocate host memory for result
    unsigned int size_C = WC * HC * 8;
    unsigned int mem_size_C = sizeof(cl_short) * size_C;
    cl_short* h_C = (cl_short*) malloc(mem_size_C);

    // create OpenCL buffer pointing to the host memory
    cl_mem h_A = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				    mem_size_A, h_A_data, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, "Error: clCreateBuffer\n");
        return ciErrNum;
    }

    // Program Setup
    size_t program_length;
    const char* header_path = shrFindFilePath("matrixMul.h", argv[0]);
    char* header = oclLoadProgSource(header_path, "", &program_length);
    if(!header)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to load the header %s!\n", header_path);
        return -1000;
    }
    const char* source_path = shrFindFilePath("matrixMul.cl", argv[0]);
    char *source = oclLoadProgSource(source_path, header, &program_length);
    if(!source)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to load compute program %s!\n", source_path);
        return -2000;
    }

    // create the program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, 
                                                    &program_length, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to create program\n");
        return ciErrNum;
    }
    free(header);
    free(source);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable -cl-fast-relaxed-math" /*-cl-denorms-are-zero -cl-single-precision-constant"*/, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then return error
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
        return ciErrNum;
    }

    // write out PTX if requested on the command line
    if(shrCheckCmdLineFlag(argc, argv, "dump-ptx") )
    {
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
    }

    // Create Kernel
    for(unsigned int i=0; i<ciDeviceCount; ++i) {
        multiplicationKernel[i] = clCreateKernel(cpProgram, "matrixMul", &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(LOGBOTH, 0, "Error: Failed to create kernel\n");
            return ciErrNum;
        }
    }
#if 0
    cl_short* reference = (cl_short*) malloc(mem_size_C);
    shrDeltaT(0);
    computeGold(reference, h_A_data, h_B_data, HA, WA, WB);
    double dSeconds = shrDeltaT(0);
    double dSize = ((double)WA * (double)HA * (double)WB * (double)HB * 8 * 8);
    shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, [CPU] Throughput = %.4f, Time = %.5f, Size = %.0f\n", 
            1.0e-9 * dSize/dSeconds, dSeconds, dSize);
#endif

    // Run multiplication on 1..deviceCount GPUs to compare improvement
    shrLog(LOGBOTH, 0, "\nRunning Computations on 1 - %d GPU's...\n", ciDeviceCount);
for(int l = 0; l < 10; l++)
    for(unsigned int k = 1; k <= ciDeviceCount; ++k) 
    {
        matrixMulGPU(k, h_A, h_B_data, mem_size_B, h_C);
    }

#if 0
    // compute reference solution
    shrLog(LOGBOTH, 0, "\nComparing results with CPU computation... \n\n");
    // check result
    shrBOOL res = Compare(reference, h_C, WC, HC);//, 1e-6f);
    shrLog(LOGBOTH, 0, "TEST %s \n\n", (1 == res) ? "PASSED" : "FAILED !!!");
    if (res != 1) 
    {
        printDiff(reference, h_C, WC, HC);
    }
#endif

    // clean up OCL resources
    clReleaseMemObject(h_A);
    for(unsigned int k = 0; k < ciDeviceCount; ++k) 
    {
        clReleaseKernel( multiplicationKernel[k] );
        clReleaseCommandQueue( commandQueue[k] );
    }
    clReleaseProgram(cpProgram);
    ciErrNum = clReleaseContext(cxGPUContext);
    if( ciErrNum != CL_SUCCESS) 
        shrLog(LOGBOTH, 0, "Error: Failed to release context: %d\n", ciErrNum);

    cl_short* referenceNeon = (cl_short*) malloc(mem_size_C);
for(int l = 0; l < 10; l++)
{
    shrDeltaT(0);
    computeNeon(referenceNeon, h_A_data, h_B_data, HA, WA, WB);
    double dSeconds = shrDeltaT(0);
    double dSize = ((double)WA * (double)HA * (double)WB * (double)HB * 8 * 8);
    shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, [CPU+NEON] Throughput = %.4f, Time = %.5f, Size = %.0f\n", 
            1.0e-9 * dSize/dSeconds, dSeconds, dSize);
}
    // check result
    shrBOOL res = Compare(referenceNeon, h_C, WC, HC);//, 1e-6f);
    shrLog(LOGBOTH, 0, "TEST %s \n\n", (1 == res) ? "PASSED" : "FAILED !!!");
    if (res != 1) 
    {
        printDiff(referenceNeon, h_C, WC, HC);
    }

    // clean up memory
    free(h_A_data);
    free(h_B_data);
    free(h_C);
    //free(reference);
    free(referenceNeon);
    
    return 0;
}

#if 0
void print(cl_short8 *data1, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      shrLog(LOGBOTH, 0, "(%d,%d) %i\n", i,j, data1[k]);
    }
  }
}
#endif

void printDiff(cl_short *data1, cl_short *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k], data2[k]);
/*
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s0, data2[k].s0);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s1, data2[k].s1);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s2, data2[k].s2);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s3, data2[k].s3);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s4, data2[k].s4);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s5, data2[k].s5);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s6, data2[k].s6);
      shrLog(LOGBOTH, 0, "(%d,%d) CPU=%i, GPU=%i \n", i,j, data1[k].s7, data2[k].s7);
*/
#if 0
      if ( fabs(data1[k] - data2[k]) < 1e-5) {
          shrLog(LOGBOTH, 0, "diff(%d,%d) CPU=%if, GPU=%if \n", i,j, data1[k], data2[k]);
          error_count++;
      }
#endif
    }
      shrLog(LOGBOTH, 0, "\n");
  }
  shrLog(LOGBOTH, 0, " \nTotal Errors = %d \n", error_count);
}
