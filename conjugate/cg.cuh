#include <stdio.h>
#include <cuda_runtime.h> 
#include <helper_cuda.h>       // helper function CUDA error checking and initialization
#include <device_launch_parameters.h>

extern int cgTest();
extern __host__ int conjugateGradient(int N, int nz, int* I, int* J, double* val, double* x, double* rhs);