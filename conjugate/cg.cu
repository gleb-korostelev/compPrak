/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /*
  * This sample implements a conjugate gradient solver on GPU
  * using CUBLAS and CUSPARSE
  *
  */

  // includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

__host__ int conjugateGradient(int N, int nz, int* I, int* J, double* val, double* x, double* rhs);

/* genTridiag: generate a random tridiagonal symmetric matrix */
inline void genTridiag(int*& I, int*& J, double*& val, int N, int& nz)
{
    nz = (N - 2) * 3 + 4;

    I = (int*)malloc(sizeof(int) * (N + 1));
    J = (int*)malloc(sizeof(int) * nz);
    val = (double*)malloc(sizeof(double) * nz);

    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (double)rand() / RAND_MAX + 10.0f;
    val[1] = (double)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i - 1] + 3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1)
        {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = (double)rand() / RAND_MAX + 10.0f;

        if (i < N - 1)
        {
            val[start + 2] = (double)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

int cgTest() {
    int N = 1048576, nz, * I = NULL, * J = NULL;
    double* val = NULL;
    double* x;
    double* rhs;

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    genTridiag(I, J, val, N, nz);

    x = (double*)malloc(sizeof(double) * N);
    rhs = (double*)malloc(sizeof(double) * N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }

    const int status = conjugateGradient(N, nz, I, J, val, x, rhs);

    return status;
}

__host__ int conjugateGradient(int N, int nz, int* I, int* J, double* val, double* x, double* rhs) {
    const double tol = 1e-9;
    const int max_iter = N;
    double a, b, na, r0, r1;
    int* d_col, * d_row;
    double* d_val, * d_x, dot;
    double* d_r, * d_p, * d_Ax;
    int k;
    double alpha, beta, alpham1;

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;

    // Otherwise pick the device with highest Gflops/s
    int devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
        deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    checkCudaErrors(cudaMalloc((void**)&d_col, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_val, nz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_x, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_r, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_p, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_Ax, N * sizeof(double)));

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseSpMatDescr_t matA = NULL;
    checkCudaErrors(cusparseCreateCsr(
        &matA, N, N, nz, d_row, d_col, d_val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseDnVecDescr_t vecx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_64F));

    /* Initialize problem data */
    cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    /* Allocate workspace for cuSPARSE */
    size_t bufferSize = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void* buffer = NULL;
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    /* Begin CG */
    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    k = 1;

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
            vecp, &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
        cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

    double rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++)
        {
            rsum += val[j] * x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    if (matA) { checkCudaErrors(cusparseDestroySpMat(matA)); }
    if (vecx) { checkCudaErrors(cusparseDestroyDnVec(vecx)); }
    if (vecAx) { checkCudaErrors(cusparseDestroyDnVec(vecAx)); }
    if (vecp) { checkCudaErrors(cusparseDestroyDnVec(vecp)); }

    //free(I);
    //free(J);
    //free(val);
    //free(x);
    //free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    printf("Test Summary:  Error amount = %f\n", err);

    return (k <= max_iter) ? 0 : 1;
}