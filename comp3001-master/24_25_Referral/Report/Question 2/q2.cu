
//implementation #1 
//float y[N], x[N], a[N*N]
void MVM_ver1(){

int i,j;

for (i=0; i<N; i++) 
 for (j=0; j<N; j++) 
 y[i]+=a[N*i+j]*x[j];

}


//multi-threading + vectorization using OpenMP
//float y[N], x[N], a[N*N]
void MVM_ver2(){

int i,j;
float tmp;

#pragma omp parallel for shared(y,a,x) private(i,j) schedule(static) 
for (i=0; i<N; i++) {
tmp=y[i];
#pragma omp simd aligned(y,x,a:64) reduction(+:tmp)
 for (j=0; j<N; j++) {
 tmp+=a[N*i+j]*x[j];
}
y[i]=tmp;
}

}


//TILE=32
// dim3 dimBlock(TILE, 1);
// dim3 dimGrid(N / TILE, 1, 1);
__global__ void mvm_ver3(float* Y, float* A, float* X) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = 0; j < N; j++)
		Y[id] += A[N * (id)+j] * X[j];

}


