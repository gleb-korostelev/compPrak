#pragma once
#include <iostream>
#include <vector>

using namespace std;

void printABCDet(int ie, int edim, double det, double* a, double* b, double* c) {
    printf("\n\n%s %d\n", "elem", ie);
    printf("%s %-.3lf\n", "det", det);
    printf("%-15s %-15s %-15s\n", "a", "b", "c");
    for (int i = 0; i < edim; i++)
    {
        printf("%-15.3lf %-15.3lf %-15.3lf\n", a[i], b[i], c[i]);
    }
}

void printStiffness(int N, double* stiffness) {
    printf("matrix %dx%d\n", N, N);
    for (int i = 0; i < N * N; i++)
    {
        printf("%-12.4lf", stiffness[i]);

        if ((i + 1) % N == 0) {
            printf("\n");
        }

    }
    printf("\n");
}

void printStiffnessF(int N, double* stiffness, double* f = NULL) {
    printf("matrix %dx%d\n", N, N);
    int j = 0;
    for (int i = 0; i < N * N; i++)
    {
        if (stiffness[i] == 0)
            printf("%-12.4s", "0");
        else
            printf("%-12.4lf", stiffness[i]);

        if ((i + 1) % N == 0) {
            if (f != NULL)
                printf("|%-12.4lf\n", f[j]);
            else
                printf("\n");
            j++;
        }

    }
    printf("\n");
}


void printSCR(int M, int N, int nz, double* val, int* I, int* J) {

    printf("\nnz=%d\n", nz);
    printf("val[%d]\nI[%d]\nJ[%d]\n", nz, M + 1, nz);
    printf("%-10s%-10s%-10s\n", "val", "I", "J");
    for (int i = 0; i < nz; i++)
    {
        if (i < M + 1) {
            printf("%-10.3lf%-10d%-10d\n", val[i], I[i], J[i]);
        }
        else {
            printf("%-10.3lf%-10s%-10d\n", val[i], " ", J[i]);
        }

    }
    printf("\n\n");
}


void resultPrint(int nsize, int ndim, double* node, double* result) {
    printf("\n\n%-7s", "i");
    for (int i = 0; i < ndim; i++)
    {
        printf("x%-9i", i);
    }
    for (int i = 0; i < ndim; i++)
    {
        printf("u%-14i", i);
    }
    printf("\n");
    for (int p = 0; p < nsize; p++) {
        printf("%-7d", p);
        for (int i = 0; i < ndim; i++)
        {
            printf("%-10.3lf", node[ndim * p + i]);
        }
        for (int i = 0; i < ndim; i++)
        {
            printf("%-15.3e", result[ndim * p + i]);
        }
        printf("\n");
    }
    printf("\n\n");


}