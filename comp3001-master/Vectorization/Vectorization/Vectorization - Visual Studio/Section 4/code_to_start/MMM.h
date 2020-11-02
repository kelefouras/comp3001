#pragma once


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>


#define N 16 //input size
 

void MMM_init();
unsigned short int MMM_default();
unsigned short int MMM_SSE();
unsigned short int MMM_AVX();
unsigned short int Compare_MMM();

extern unsigned short int equal(float const a, float const b); //used extern as this is defined in MVM.h

#define EPSILON 0.01


