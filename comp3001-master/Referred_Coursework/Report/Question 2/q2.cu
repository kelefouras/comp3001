//implementation #1 

for (int i = 0; i < N; i++) 

for (int j = 0; j < N; j++) 

for (int k = 0; k < N; k++) 

C[N * i + j] += A[N * i + k] * B[N * k + j]; 

 

 

//implementation #2 

__declspec(align(64)) float C[N*N], A[N*N], B[N*N]; 

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

 

//#Implementation #3 

//use dim3 dimBlock(16, 16, 1); 
//use dim3 dimGrid(N/16, N/16, 1); 

__global__ void mmm_ver1(float *C,float *A, float*B) { 

 

float tmp = 0.0; 

 

int i = blockIdx.x * blockDim.x + threadIdx.x; //i loop has been parallelized 

int j = blockIdx.y * blockDim.y + threadIdx.y; //j loop has been parallelized 

  

for (int k = 0; k < N; k++) { 

tmp += A[N * i + k] * B[N * k + j]; 

} 

 

C[N * i + j] = tmp; 

 

} 

 

//implementation #4 
//use dim3 dimBlock(16, 16, 1); 
//use dim3 dimGrid(N/16, N/16, 1); 

__global__ void mmm_tiled(float* C, float* A, float* B) { 

 

__shared__ float aa[16][16]; 

__shared__ float bb[16][16]; 

float tmp = 0.0; 

int k, m; 

 

int row_A = 16 * blockIdx.y + threadIdx.y; 

int col_B = blockIdx.x * 16 + threadIdx.x; 

 

for (m = 0; m < N / 16; m++) { 

aa[threadIdx.y][threadIdx.x] = A[N * (row_A)+(m * 16 + threadIdx.x)]; 

bb[threadIdx.y][threadIdx.x] = B[N * (m * TILE + threadIdx.y) + (col_B)]; 

 

__syncthreads(); 

 

for (k = 0; k < 16; k ++) { 

tmp += aa[threadIdx.y][k] * bb[k][threadIdx.x]; 

} 

__syncthreads(); 

} 

C[N * row_A + col_B] = tmp; 

} 



