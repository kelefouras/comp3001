
//implementation #1 
	for (i = 0; i < N; i++) {
	  for (j = 0; j < N; j++) {
	    Y[i] += A[N * i + j] * X[j];
				}
		        }

 

//implementation #2 

__declspec(align(64)) float Y[N], test[N], A[N*N], X[N]; 

#pragma omp parallel for private(i, j, tmp)
	for (i = 0; i < N; i++) {
		tmp = 0.0;
#pragma omp simd reduction(+:tmp) aligned(Y,A,X:64)
		for (j = 0; j < N; j++) {
		  tmp += A[N * i + j] * X[j];
					}
		Y[i] = tmp;
		}
	


 

//#Implementation #3 
//use dim3 dimBlock(256, 1);
//use dim3 dimGrid(N / 256, 1, 1);

__global__ void mvm_ver3(float *Y,float *A, float*X) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = 0; j < N; j++)
		Y[id] += A[N * (id)+j] * X[j];
	

}

 



