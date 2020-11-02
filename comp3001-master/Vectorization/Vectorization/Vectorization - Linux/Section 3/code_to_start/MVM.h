#pragma once

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>	/* for uint64 definition */
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>

//MVM initialization 
#define M 100000


void initialization_MVM();

unsigned short int MVM_default();
unsigned short int MVM_SSE();
unsigned short int MVM_AVX();
unsigned short int Compare_MVM();
unsigned short int equal(float const a, float const b);

#define EPSILON 0.01


