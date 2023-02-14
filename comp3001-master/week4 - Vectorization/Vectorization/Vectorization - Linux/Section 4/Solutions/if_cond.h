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
#define N_IF 512


void if_cond_init();
unsigned short int if_cond_default();
unsigned short int if_cond_SSE();
unsigned short int if_cond_AVX();
unsigned short int Compare_if_cond();


extern unsigned short int equal(float const a, float const b); //used extern as this is defined in MVM.h

#define EPSILON 0.01


