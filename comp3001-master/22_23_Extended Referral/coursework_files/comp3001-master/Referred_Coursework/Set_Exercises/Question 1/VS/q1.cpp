/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <Windows.h>
#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

#include <omp.h>


#define TIMES 1
#define N 4096


__declspec(align(64)) float  X[N], Y[N] ;
__declspec(align(64)) float A[N][N], Atr[N * N] ;


int main() {


    int i, j, count = 0;
float alpha=0.234f;


    //the following command pins the current process to the 1st core
    //otherwise, the OS tongles this process between different cores
    BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
    if (success == 0) {
        //cout << "SetProcessAffinityMask failed" << endl;
        printf("\nSetProcessAffinityMask failed\n");
        system("pause");
        return -1;
    }

    //-------------------initialize ---------------------------
    for (i = 0; i < N; i++) {
        Y[i] = 0.0;
        X[i] = (float) ((i % 99) / 3);
    }


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            A[i][j] = (float) (((i + j) % 99) / 3);


    //define the timers measuring execution time
    clock_t start_1, end_1; //ignore this for  now
    start_1 = clock();


    //-------------------main kernel ---------------------------
    for (int it = 0; it < TIMES; it++)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                Y[i] += alpha * A[i][j] * X[j];





    end_1 = clock(); //end the timer
    printf("elapsed time = %f seconds\n", (float)(end_1-start_1) / CLOCKS_PER_SEC);


    printf("\n The first and last values are %f %f\n", Y[0], Y[N - 1]);


    return 0;
}
