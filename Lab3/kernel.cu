
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_OF_ITERATIONS 3
#define N 4
#define MAX_NUMBER_THREADS 1024

cudaError_t drumWithCuda();

__global__ void drumKernelinner(double *u, double *u1, double *u2)
{
	double p = 0.5;
	double nConst = 0.0002;
	double G = 0.75;

	if ((threadIdx.x % N != 0) && (threadIdx.x % N != N-1)) {
		if (threadIdx.x > N && threadIdx.x < ((N * N) - N)) {
			u[threadIdx.x] = (p * (u1[threadIdx.x - N] + u1[threadIdx.x + N] + u1[threadIdx.x - 1] + u1[threadIdx.x + 1]
				- 4 * u1[threadIdx.x]) + (2 * u1[threadIdx.x]) - ((1-nConst) * u2[threadIdx.x])) / (1 + nConst);
		}
	}
}

__global__ void drumKernelsides(double* u, double* u1, double* u2)
{
	double p = 0.5;
	double nConst = 0.0002;
	double G = 0.75;

	if (threadIdx.x < N) {
		if (threadIdx.x % N != 0 && threadIdx.x % N != N - 1) {
			u[threadIdx.x] = G * u[N + threadIdx.x];
		}
	}
	if (threadIdx.x % N == 0) {
		if ((!(threadIdx.x < N)) && (threadIdx.x < ((N * N) - N))) {
			u[threadIdx.x] = G * u[threadIdx.x + 1];
		}
	}
	if (threadIdx.x % N == N - 1) {
		if ((!(threadIdx.x < N)) && (threadIdx.x < ((N * N) - N))) {
			u[threadIdx.x] = G * u[threadIdx.x - 1];
		}
	}
	if (threadIdx.x > ((N * N) - N)) {
		if (threadIdx.x % N != 0 && threadIdx.x % N != N - 1) {
			u[threadIdx.x] = G * u[threadIdx.x - N];
		}
	}
}

__global__ void copyCurrToPrev(double* u, double* u1, double* u2) {
	u2[threadIdx.x] = u1[threadIdx.x];
	u1[threadIdx.x] = u[threadIdx.x];
}


int main()
{
	double const p = 0.5;
	double const nConst = 0.0002;
	double G = 0.75;
	double u[N * N] = { 0.0 };
	double u1[N * N] = { 0.0 }; 
	double u2[N * N] = { 0.0 };
	u1[(N*(N/2)) + N/2] = 1.0;
	for (int z = 0; z < NUM_OF_ITERATIONS; z++) {
		for (int i = 1; i <= N-2; i++) {
			for (int j = 1; j <= N-2; j++) {
				u[(i*N) + j] = (p * (u1[((i-1) * N) + j] + u1[((i+1) * N) + j] + u1[(i * N) + (j-1)] + u1[(i * N) + (j + 1)] - 4 * u1[(i*N) + j]) +
					2 * u1[(i*N)+j] - (1 - nConst) * u2[(i * N) + j]) / (1 + nConst);
			}
		}
		for (int i = 1; i <= N-2; i++) {
			u[i] = G * u[N+i];
			u[(N*(N - 1))+i] = G * u[(N * (N - 2)) + i];
			u[i*N] = G * u[(i*N) + 1];
			u[(i*N) + (N - 1)] = G * u[(i * N) + (N - 2)];
		}
		u[0] = G * u[N];
		u[(N - 1)*N] = G * u[(N - 1) * N];
		u[N - 1] = G * u[N - 2];
		u[N*(N - 1) + (N - 1)] = G * u[N * (N - 1) + (N - 2)];
		memcpy(&u2, &u1, (N * N) * sizeof(double));
		memcpy(&u1, &u, (N * N) * sizeof(double));
		printf("%f \n", u[(N * (N / 2)) + N / 2]);
	}

	drumWithCuda();

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t drumWithCuda()
{
    cudaError_t cudaStatus;
	double *u, *u1, *u2;
	double G = 0.75;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	cudaStatus = cudaMallocManaged((void**)& u, (N * N) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of u failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& u1, (N * N) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of u1 failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& u2, (N * N) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of u2 failed!");
		goto Error;
	}

	memset(&u[0], 0, (N * N) * sizeof(double));
	memset(&u1[0], 0, (N * N) * sizeof(double));
	memset(&u2[0], 0, (N * N) * sizeof(double));
	
	u1[(N * (N / 2)) + N / 2] = 1.0;
	
	int numBlocks = 1;
	int threadsPerBlock = 16;
	if (N == 512) {
		numBlocks = ((N + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS) + 1;
		threadsPerBlock = ((N + (numBlocks - 1)) / numBlocks);
	}
	for (int x = 0; x < NUM_OF_ITERATIONS; x++) {
		// Launch a kernel on the GPU with one thread for each element.
		drumKernelinner << <numBlocks, threadsPerBlock >> > (u, u1, u2);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "drumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		//Update Sides
		drumKernelsides << <1, 16 >> > (u, u1, u2);
		//or
		//drumKernelinner512 << < >> > (u, u1, u2);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "drumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		//update corners
		u[0] = G * u[N];
		u[(N - 1)*N] = G * u[(N - 1) * N];
		u[N - 1] = G * u[N - 2];
		u[N*(N - 1) + (N - 1)] = G * u[N * (N - 1) + (N - 2)];

		//Copy current values to u1 and u1 to u2
		copyCurrToPrev << <1, 16 >> > (u, u1, u2);
		//update To
		//copyCurrToPrev << <numBlocks, threadsPerBlock >> > (u, u1, u2);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "drumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		printf("%f \n", u[(N * (N / 2)) + N / 2]);
	}  

Error:
	cudaFree(u);
	cudaFree(u1);
	cudaFree(u2);

    return cudaStatus;
}
