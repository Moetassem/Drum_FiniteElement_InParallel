#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 4 //for sequential only
#define MAX_NUMBER_THREADS 1024

cudaError_t drumWithCuda(int userN, int userNumOfBlocks, int userNumOfThreadsPerBlock, int userNumOfIterations);

__global__ void drumKernelinner(double *u, double *u1, double *u2, int threadsPerBlock, int numOfElementsPerBlock, int userN)
{
	for (int i = 0; i < numOfElementsPerBlock / threadsPerBlock; i++) {
		int j = (threadIdx.x + threadsPerBlock * i) + (blockIdx.x * numOfElementsPerBlock);
		double p = 0.5;
		double nConst = 0.0002;

		if ((j % userN != 0) && (j % userN != userN - 1)) {
			if (j > userN && j < ((userN * userN) - userN)) {
				u[j] = (p * (u1[j - userN] + u1[j + userN] + u1[j - 1] + u1[j + 1]
					- 4 * u1[j]) + (2 * u1[j]) - ((1 - nConst) * u2[j])) / (1 + nConst);
			}
		}
	}
}

__global__ void drumKernelsides(double* u, double* u1, double* u2, int threadsPerBlock, int numOfElementsPerBlock, int userN)
{
	for (int i = 0; i < numOfElementsPerBlock / threadsPerBlock; i++) {
		int j = (threadIdx.x + threadsPerBlock * i) + (blockIdx.x * numOfElementsPerBlock);
		double G = 0.75;

		if (j < userN) {
			if (j % userN != 0 && j % userN != userN - 1) {
				u[j] = G * u[userN + j];
			}
		}
		if (j % userN == 0) {
			if ((!(j < userN)) && (j < ((userN * userN) - userN))) {
				u[j] = G * u[j + 1];
			}
		}
		if (j % userN == userN - 1) {
			if ((!(j < userN)) && (j < ((userN * userN) - userN))) {
				u[j] = G * u[j - 1];
			}
		}
		if (j > ((userN * userN) - userN)) {
			if (j % userN != 0 && j % userN != userN - 1) {
				u[j] = G * u[j - userN];
			}
		}
	}
}

__global__ void copyCurrToPrev(double* u, double* u1, double* u2, int threadsPerBlock, int numOfElementsPerBlock) {
	for (int i = 0; i < numOfElementsPerBlock / threadsPerBlock; i++) {
		int j = (threadIdx.x + threadsPerBlock * i) + (blockIdx.x * numOfElementsPerBlock);
		u2[j] = u1[j];
		u1[j] = u[j];
	}
}


int main(int argc, char* argv[])
{
	char* SeqOrPar = nullptr;
	int userN = 0;
	int userNumOfBlocks = 0;
	int userNumOfThreadsPerBlock = 0;
	int userNumOfIterations = 0;
	if (argc < 3 || argv[1] == NULL || argv[2] == NULL ||
		argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
			printf("Lab3.exe <Sequential> <Number of Iterations> <Drum size is 4x4>\n" 
				"OR \n <Parallel> <Number of Iterations> + Optional in this order: <Number of Blocks> <Number of Threads Per Block> <Drum Size (N)>\n"
				"E.x: Lab3.exe Parallel 12 64 64 512\n");
		return 0;
	}
	else {
		if (argv[1] != NULL) {
			SeqOrPar = argv[1];
		}
		if (argv[2] != NULL) {
			userNumOfIterations = atoi(argv[2]);
		}
		if (argv[3] != NULL) {
			userNumOfBlocks = atoi(argv[3]);
			if (argv[4] != NULL) {
				userNumOfThreadsPerBlock = atoi(argv[4]);
				if (argv[5] != NULL) {
					userN = atoi(argv[5]);
				}
			}
		}
	}

	if (!strcmp(SeqOrPar, "Sequential")) {
		double const p = 0.5;
		double const nConst = 0.0002;
		double G = 0.75;
		double u[N * N] = { 0.0 };
		double u1[N * N] = { 0.0 }; 
		double u2[N * N] = { 0.0 };
		u1[(N*(N/2)) + N/2] = 1.0;
		for (int z = 0; z < userNumOfIterations; z++) {
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
			printf("u[2,2] at iteration %d: %f \n", z+1, u[(N * (N / 2)) + N / 2]);
		}
	}
	else {
		drumWithCuda(userN, userNumOfBlocks, userNumOfThreadsPerBlock, userNumOfIterations);
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t drumWithCuda(int userN, int userNumOfBlocks, int userNumOfThreadsPerBlock, int userNumOfIterations)
{
    cudaError_t cudaStatus;
	clock_t start_t, end_t;
	double *u, *u1, *u2;
	double G = 0.75;

	if (userN == NULL) {
		userN = 512;
		printf("Drum size assumed to be 512x512 \n");
	}

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	//Mallocing u, u1, u2
	cudaStatus = cudaMallocManaged((void**)& u, (userN * userN) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of u failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& u1, (userN * userN) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of u1 failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& u2, (userN * userN) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of u2 failed!");
		goto Error;
	}

	//initializing to 0
	memset(&u[0], 0, (userN * userN) * sizeof(double));
	memset(&u1[0], 0, (userN * userN) * sizeof(double));
	memset(&u2[0], 0, (userN * userN) * sizeof(double));
	
	//
	u1[(userN * (userN / 2)) + userN / 2] = 1.0;
	
	int numBlocks = 0;
	int threadsPerBlock = 0;
	int numOfElementsPerBlock = 0;

	//Verifing and validating the user input 
	if (userN == 4) {
		numBlocks = 1;
		threadsPerBlock = 16;
	}
	else {
		userN = 512;
		numBlocks = userNumOfBlocks;
		threadsPerBlock = userNumOfThreadsPerBlock;
		if ((numBlocks*threadsPerBlock > userN*userN) || threadsPerBlock > MAX_NUMBER_THREADS) {
			printf("Using more threads or blocks than needed\nRan with maximum number of blocks and threads per block \n");
			threadsPerBlock = MAX_NUMBER_THREADS;
			numBlocks = 16;
		}
		if (userNumOfBlocks == 0 || userNumOfThreadsPerBlock == 0) {
			numBlocks = ((userN * userN + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
			threadsPerBlock = ((userN * userN + (numBlocks - 1)) / numBlocks);
		}
	}
	numOfElementsPerBlock = userN * userN / numBlocks;
	printf("Drum Dimension: %dx%d\n", userN, userN);
	printf("Number of Blocks: %d\n", numBlocks);
	printf("Number of Threads Per Block: %d\n", threadsPerBlock);
	printf("Number of Elements Per Thread: %d\n", (userN * userN) / (numBlocks*threadsPerBlock));

	start_t = clock();
	for (int x = 0; x < userNumOfIterations; x++) {
		// Launch a kernel on the GPU with one thread for each element.
		drumKernelinner << <numBlocks, threadsPerBlock >> > (u, u1, u2, threadsPerBlock, numOfElementsPerBlock, userN);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "drumKernelinner launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching drumKernelinner!\n", cudaStatus);
			goto Error;
		}

		//Update Sides
		drumKernelsides << <numBlocks, threadsPerBlock>> > (u, u1, u2, threadsPerBlock, numOfElementsPerBlock, userN);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "drumKernelSides launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching drumKernelSides!\n", cudaStatus);
			goto Error;
		}

		//update corners
		u[0] = G * u[userN];
		u[(userN - 1)* userN] = G * u[(userN - 1) * userN];
		u[userN - 1] = G * u[userN - 2];
		u[userN *(userN - 1) + (userN - 1)] = G * u[userN * (userN - 1) + (userN - 2)];

		//Copy current values to u1 and u1 to u2
		copyCurrToPrev << <numBlocks, threadsPerBlock>> > (u, u1, u2, threadsPerBlock, numOfElementsPerBlock);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "copyCurrToPrev launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyCurrToPrev!\n", cudaStatus);
			goto Error;
		}
		printf("u[256,256] at iteration %d: %f \n", x+1, u[(userN * (userN / 2)) + userN / 2]);
	}
	end_t = clock();
	printf("\n time taken: %d \n", ((end_t - start_t)));

Error:
	//BE FREE MY LOVLIES
	cudaFree(u);
	cudaFree(u1);
	cudaFree(u2);

    return cudaStatus;
}
