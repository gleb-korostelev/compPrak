
#include<algorithm>
#include <string>
#include <vector>
#include <array>
#include <cctype>
#include <functional>   // std::plus // isinf()
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "helper.h";
#include "cg.cuh";

using namespace std;

typedef vector<int> Elem;

//typedef v2 = *double
//typedef array<double, 2> v2d;
//typedef array<int, 3> v3i;

constexpr int edim = 3;

constexpr int ndim = 2;

constexpr int blockSize = ndim * ndim;

const int elem[9] =
{
     0,  1,  3,
     1,  4,  3,
     0,  3,  2,
};
const int esize = sizeof(elem) / sizeof(int) / edim;

double node[10] = // [cm]
{
    -4.0,   0.0,
    -2.0,   0.0,
    -5.0,   1.7,
    -3.0,   1.7,
    -1.0,   1.7,
};
const int nsize = sizeof(node) / sizeof(double) / ndim;

const double elastic[2] = { 20.6, 0.3 }; // { [MN/cm2], [1]}

const double thickness = 1.0; // [cm]

double load[10] = // [MN]
{
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
   -3.0e-3,
    0.0,
   -3.0e-3,
    0.0,
    0.0,
};

double bound[10] = { // [cm]
    0.0,
    0.0,
    0.0,
    0.0,
    INFINITY,
    INFINITY,
    INFINITY,
    INFINITY,
    INFINITY,
    INFINITY,
};
//bound[ndim * 2 + 0]

void getA(int* P, double* result) {// ai = eijm xj ym; a0 = x1 y2 - x2 y1; a1 = x2 y0 - x0 y2; a2 = x0 y1 - x1 y0;
    
    result[0] = node[P[1] * ndim + 0] * node[P[2] * ndim + 1] - node[P[2] * ndim + 0] * node[P[1] * ndim + 1];
    //result[1] = ;
    //result[2] = ;

}

void getB(int* P, double* result) {// bj = eijm 1i ym; b0 = y1 - y2; b1 = y2 - y0; b2 = y0 - y1;

    result[0] = node[P[1] * ndim + 1] - node[P[2] * ndim + 1];
    //result[1] = ;
    //result[2] = ;

}


void getC(int* P, double* result) {// cm = eijm 1i xj; c0 = x2 - x1; c1 = x0 - x2; c2 = x1 - x0;

    result[0] = node[P[2] * ndim + 0] - node[P[1] * ndim + 0];
    //result[1] = ;
    //result[2] = ;

}


void getDet(double* a, double& det) {
    // det = ;
}

void blockStressStiffness(double det, double bp, double bq, double cp, double cq, double* block) {

    double v = elastic[1];
    double E = elastic[0];
    double k = (1 - v) / 2.0;
    double m = thickness * E / (1 - v * v) / 2 / det;

    block[0 * ndim + 0] = (    bp * bq + k * cp * cq) * m;
 // block[0 * ndim + 1] = ;

}

void applyKinematic(int N, double* u, double* f, double* stiffness) {
    for (int p = 0; p < N; p++)
    {
        for (int q = 0; q < N; q++)
        {
            // 
        }
    }
    printf("\nKinematic boundary condition is applyed\n\n");
}

int kinematicTest() {// Л.Сегерлинд. Применение метода конечных элементов стр. 110-112
    const int N = 5;
    double u[N] = {
        150,
        INFINITY,
        INFINITY,
        INFINITY,
        40,
    }; 
    double f[N] = {
        500,
        2000,
        1000,
        2000,
        900,
    };
    double stiffness[N * N] = {
        55, -46,   4,   0,   0,
       -46, 140, -46,   0,   0,
         4, -46, 110, -46,   4,
         0,   0, -46, 142, -46,
         0,   0,   4, -46,  65,
    };
    printStiffnessF(N, stiffness, f);
    applyKinematic(N, u, f, stiffness);
    printStiffnessF(N, stiffness, f);

    double sum = 
        stiffness[0 * N + 0] -  55.0 + f[0] - 8250 +
        stiffness[1 * N + 1] - 140.0 + stiffness[1 * N + 2] +  46.0 + f[1] - 8900 +
        stiffness[2 * N + 1] +  46.0 + stiffness[2 * N + 2] - 110.0 + stiffness[2 * N + 3] + 46.0 + f[2] - 240 +
        stiffness[3 * N + 2] +  46.0 + stiffness[3 * N + 3] - 142.0 + f[3] - 3840 +
        stiffness[4 * N + 4] - 65.0 + f[4] - 2600;

    return sum == 0.0 ? 0 : 1;

}

int sumStiffnessTest(int N, double* stiffness) {
    const double tol = 1e-9;
    double sum = 0;
    
    //


    printf("\n\nSum stiffness is %5.9e\n\n", sum);

    return abs(sum) < tol ? 0 : 1;
}

// Преобразовать вектор-матрицу жёсткости в формат CSR https://docs.nvidia.com/cuda/cusparse/index.html#csr-format
void transformMatrixToCsr(int M, int N, int& nz, double* stiffness, double*& _val, int*& _I, int*& _J) {
    
    vector<double> val;
    vector<int> I(N + 1);
    vector<int> J;

    nz = 0;
    int icounter = 0;
    
    for (int i = 0; i < M; i++)
    {
        // I[icounter] =
        icounter++;
        for (int j = 0; j < N; j++)
        {
           
            //if (stiffness == 0) continue;

            //val.push_back;
            //J.push_back
            nz++;

        }

    }
    // I[icounter] = 

    _I = (int*)malloc((M + 1) * sizeof(int));
    _J = (int*)malloc(nz * sizeof(int));
    _val = (double*)malloc(nz * sizeof(double));

    for (int i = 0; i < nz; i++)
    {
        if (i < M + 1) {
            _I[i] = I[i];
        }
        _val[i] = val[i];
        _J[i] = J[i];
    }
    
}

void transformMatrixTest() {
    const int M = 4;
    const int N = 5;

    double stiffness[M*N] = {
        1.0, 4.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 3.0, 0.0, 0.0,
        5.0, 0.0, 0.0, 7.0, 8.0,
        0.0, 0.0, 9.0, 0.0, 6.0,
    };

    double* val = NULL;
    int* I = NULL;
    int* J = NULL;
    int nz;
    transformMatrixToCsr(M, N, nz, stiffness, val, I, J);
    printSCR(M, N, nz, val, I, J);

}

int main(int argc, char** argv)
{
    int N = nsize * ndim;
    
    double* globalStiffness = (double*)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++)
    {
        globalStiffness[i] = 0;

    }

    printStiffness(N, globalStiffness);

    //for (int ie = 0; ie < esize; ie++)
    //{
    //   
    //    double a[3], b[3], c[3];

    //    int P[3] = {
    //        elem[ie * edim + 0],
    //        ,
    //        ,
    //    };

    //    int Q[3] = {
    //        P[0],
    //        ,
    //        ,
    //    };

    //    double det = 0;
    //    getA(P, a);
    //    getB(P, b);
    //    getC(P, c);
    //    getDet(a, det);

    //    printABCDet(ie, edim, det, a, b, c);

    //    
    //    
    //    for (int p = 0; p < edim; p++)
    //    {

    //        for (int q = 0; q < edim; q++)
    //        {
    //   
    //            double block[ndim * ndim];

    //            blockStressStiffness(det, b[p], b[q], c[p], c[q], block);

    //            printf("\nPQ[%d%d] pq[%d%d]\n", P[p], Q[q], p, q);
    //            printStiffness(ndim, block);

    //            for (int i = 0; i < ndim; i++)
    //            {
    //                for (int j = 0; j < ndim; j++)
    //                {
    //                    
    //                    globalStiffness[(P[p] * ndim + i) * N + (...)] += block[...];
    //                }
    //            }
    //        }
    //    }

    //    
    //}

    int status = sumStiffnessTest(N, globalStiffness);

    if (status != 0) {
        printf("\nSum stiffness error\n");
        int exit(status);
    }
    else {
        printf("\nSum stiffness ok\n");
    }

    //printStiffnessF(N, globalStiffness, load);
    //applyKinematic(N, bound, load, globalStiffness);
    //printStiffnessF(N, globalStiffness, load);

    //status = kinematicTest();

    if (status != 0) {
        printf("\nKinematic error\n");
        int exit(status);
    }
    else {
        printf("\nKinematic test ok\n");
    }

    //transformMatrixTest();

    double* val = NULL;
    int* I = NULL;
    int* J = NULL;
    int nz;

    //transformMatrixToCsr(N, N, nz, globalStiffness, val, I, J);
    //printSCR(N, N, nz, val, I, J);

    double* result = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        result[i] = 0.0;
    }

    // //status = cgTest();
    //status = conjugateGradient(N, nz, I, J, val, result, load);

    //resultPrint(nsize, ndim, node, result);

    int exit(status);


}
