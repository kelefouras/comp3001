//implementation #1 

void mmm_ver1() {
	__m256 ymm0, ymm1, ymm2, acc0,acc1,acc2,acc3;
	__m128 xmm1, xmm2;
	int i, j, k;
	float temp;
	
	if (N%4!=0){
	printf("\n works for multiples of 4 only - padding code is needed");
	return -1;
	}
	
	//B[][] array needs to be written in maim memory column-wise and not row-wise
//SIMD hardware can load only consecutive main memory locations and the arrays are written in main memory row-wise
//thus, this loop kernels copies the data from B[][] and stores them to Btranspose[][]
//this loop kernel can be implemented using SSE too, thus boosting performance
	for (j = 0; j != N; j++)
		for (k = 0; k != N; k++) {
			Btranspose[k][j] = B[j][k];
		}


	for (i = 0; i != N; i++)
		for (j = 0; j != N; j+=4) {//reg.blocking with a factor of 4
			acc0 = _mm256_setzero_ps();//j
			acc1 = _mm256_setzero_ps();//j+1
			acc2 = _mm256_setzero_ps();//j+2
			acc3 = _mm256_setzero_ps();//j+3
			
			for (k = 0; k != ((N / 8) * 8); k += 8) {
				ymm1 = _mm256_load_ps(&A[i][k]);
				
				ymm2 = _mm256_load_ps(&Btranspose[j][k]); //j
				acc0 = _mm256_fmadd_ps(ymm1, ymm2, acc0);
				
				ymm2 = _mm256_load_ps(&Btranspose[j+1][k]); //j+1
				acc1 = _mm256_fmadd_ps(ymm1, ymm2, acc1);
				
				ymm2 = _mm256_load_ps(&Btranspose[j+2][k]); //j+2
				acc2 = _mm256_fmadd_ps(ymm1, ymm2, acc2);
				
				ymm2 = _mm256_load_ps(&Btranspose[j+3][k]); //j+3
				acc3 = _mm256_fmadd_ps(ymm1, ymm2, acc3);
			}

			ymm2 = _mm256_permute2f128_ps(acc0, acc0, 1);//j
			acc0 = _mm256_add_ps(acc0, ymm2);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			xmm2 = _mm256_extractf128_ps(acc0, 0);
			_mm_store_ss(&C[i][j], xmm2);
			
			ymm2 = _mm256_permute2f128_ps(acc1, acc1, 1);//j+1
			acc0 = _mm256_add_ps(acc1, ymm2);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			xmm2 = _mm256_extractf128_ps(acc0, 0);
			_mm_store_ss(&C[i][j+1], xmm2);
			
			ymm2 = _mm256_permute2f128_ps(acc2, acc2, 1);//j+2
			acc0 = _mm256_add_ps(acc2, ymm2);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			xmm2 = _mm256_extractf128_ps(acc0, 0);
			_mm_store_ss(&C[i][j+2], xmm2);
			
			ymm2 = _mm256_permute2f128_ps(acc3, acc3, 1);//j+3
			acc0 = _mm256_add_ps(acc3, ymm2);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			acc0 = _mm256_hadd_ps(acc0, acc0);
			xmm2 = _mm256_extractf128_ps(acc0, 0);
			_mm_store_ss(&C[i][j+3], xmm2);

		}

	return 0;
}

 

//implementation #2 
//__declspec(align(64)) float C[N*N], A[N*N], B[N*N]; 

void mmm_ver2(){

#pragma omp parallel  

{ 

  #pragma omp for private(i, j, k, tmp) 

for (i = 0; i < N; i++) { 

for (j = 0; j < N; j++) { 

tmp = 0.0; 

#pragma omp simd reduction(+:tmp) aligned(C,A,B:64) 

for (k = 0; k < N; k++) { 

tmp += A[N * i + k] * B[N * k + j]; 

} 

C[N * i + j] = tmp; 

} 

} 
} 

}
 

//#Implementation #3 
//dim3 dimBlock(16, 16, 1);
//use dim3 dimGrid(N/32, N/32, 1); 
__global__ void mmm_ver3(float* C, float* A, float* B) {

	__shared__ float aa1[16][16];
	__shared__ float bb1[16][16];
	__shared__ float aa2[16][16];
	__shared__ float bb2[16][16];

	float tmp0 = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
	int k, m;

	int row_A = N * (32 * blockIdx.y + threadIdx.y);
	int col_B = blockIdx.x * 32 + threadIdx.x;


	for (m = 0; m < N / 16; m++) {
		//initialize the shared arrays
		aa1[threadIdx.y][threadIdx.x] = A[(row_A)+m * 16 + threadIdx.x];
		aa2[threadIdx.y][threadIdx.x] = A[(row_A + N * 16) + (m * 16 + threadIdx.x)];
		bb1[threadIdx.y][threadIdx.x] = B[N * (m * 16 + threadIdx.y) + (col_B)];
		bb2[threadIdx.y][threadIdx.x] = B[N * (m * 16 + threadIdx.y) + (col_B)+16];

		__syncthreads();//all threads wait until the arrays are initialized.

		for (k = 0; k < 16; k++) {
			//each thread multiplies 2 sub-rows by 2 sub-columns. each thread uses four registers for storing the intermediate results
			tmp0 += aa1[threadIdx.y][k] * bb1[k][threadIdx.x];
			tmp1 += aa1[threadIdx.y][k] * bb2[k][threadIdx.x];
			tmp2 += aa2[threadIdx.y][k] * bb1[k][threadIdx.x];
			tmp3 += aa2[threadIdx.y][k] * bb2[k][threadIdx.x];
		}

		__syncthreads();//all threads wait until all the multiplications have finished 
	}
	C[row_A + col_B] = tmp0;
	C[row_A + col_B + 16] = tmp1;
	C[row_A + N * 16 + col_B] = tmp2;
	C[row_A + N * 16 + col_B + 16] = tmp3;


}


 

//implementation #4 
//use dim3 dimBlock(16, 16, 1); 
//use dim3 dimGrid(N/16, N/16, 1); 
__global__ void mmm_ver4(float* C, float* A, float* B) { 

 

__shared__ float aa[16][16]; 

__shared__ float bb[16][16]; 

float tmp = 0.0; 

int k, m; 

 

int row_A = 16 * blockIdx.y + threadIdx.y; 

int col_B = blockIdx.x * 16 + threadIdx.x; 

 

for (m = 0; m < N / 16; m++) { 

aa[threadIdx.y][threadIdx.x] = A[N * (row_A)+(m * 16 + threadIdx.x)]; 

bb[threadIdx.y][threadIdx.x] = B[N * (m * 16 + threadIdx.y) + (col_B)]; 

 

__syncthreads(); 

 

for (k = 0; k < 16; k ++) { 

tmp += aa[threadIdx.y][k] * bb[k][threadIdx.x]; 

} 

__syncthreads(); 

} 

C[N * row_A + col_B] = tmp; 

} 



