#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <driver_types.h>

inline void checkCudaErrors(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
		exit(EXIT_FAILURE);
    }
}


void printDeviceProp(const cudaDeviceProp &prop)//cudaPointerAttributes
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %lu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %lu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %lu.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %lu.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %lu.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
    printf("unifiedAddressing : %d.\n", prop.unifiedAddressing);
}
 
bool InitCUDA()
{
    //used to count the device numbers
    int count;
 
    // get the cuda device count
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
 
    // find the device >= 1.X
    int i;
    for (i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                printDeviceProp(prop);
                break;
            }
        }
    }
 
    // if can't find the device
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

	sleep(10);

    // set cuda device
    cudaSetDevice(i);

	sleep(10);

    return true;
}

int main_v1(int argc, char const *argv[])
{
    if (InitCUDA()) {
        printf("CUDA initialized.\n");
    }
 
    return 0;
}


// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main_v2()
{
    int N = 100000000;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
	printf("set device 0\n");
    cudaSetDevice(0);
	printf("set device 0 done\n");

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	printf("run on device 0 done\n");
	sleep(3);

	printf("try to run on devcie 1\n");

    // Initialize input vectors
    cudaSetDevice(1);
	printf("set device 1 done\n");

    // Allocate vectors in device memory
    float* d_A_1;
    cudaMalloc(&d_A_1, size);
    float* d_B_1;
    cudaMalloc(&d_B_1, size);
    float* d_C_1;
    cudaMalloc(&d_C_1, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A_1, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_1, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_1, d_B_1, d_C_1, N);

	printf("run on device 1 done\n");
	sleep(3);

    // Free device memory
    cudaFree(d_A_1);
    cudaFree(d_B_1);
    cudaFree(d_C_1);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

	printf("done\n");
}

int main_cudafree()
{
	cudaFree(0);
	sleep(5);
	printf("run on device 1 start\n");
	cudaSetDevice(1);
	printf("run on device 1 done\n");
	sleep(5);
	printf("run on device 1 free start\n");
	cudaFree(0);
	printf("run on device 1 free end\n");
	sleep(5);
	printf("run on device 0 start\n");
	cudaSetDevice(0);
	printf("run on device 0 done\n");
	sleep(5);
	printf("run on device 0 cudaMalloc start\n");
    float* d_A_0;
//    cudaMalloc(&d_A_0, 100);
	size_t v_free, v_total;
//	cudaMemGetInfo(&v_free, &v_total);
	printf("run on device 0, cudamemgetinfo:free:%lu, total:%lu\n", v_free, v_total);
//	printf("run on device 0 cudaMalloc end\n");
	sleep(5);
	printf("run on device 1 start\n");
	cudaSetDevice(1);
	printf("run on device 1 done\n");
	sleep(5);
	printf("run on device 1 cudaMalloc start\n");
    float* d_A_1;
    //cudaMalloc(&d_A_1, 100);
	cudaMemGetInfo(&v_free, &v_total);
	printf("run on device 1, cudamemgetinfo:free:%lu, total:%lu\n", v_free, v_total);
	//printf("run on device 1 cudaMalloc end\n");
	return 0;
}

void print_array(int* array, int size) {
	int i;
	for (i = 0; i < size; i++) {
	    printf("%x ", array[i]);
	}
	printf("\n");
}

__global__ void VecAdd2(int* A, int* B, int* C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main_v3()
{
//    int N = 0x10000000;
    int N = 4;

    int size = N * sizeof(int);
	
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);
	memset(h_A, 0x11111111, size);
	memset(h_B, 0x22222222, size);
	int all_card = 1;
	
	for (int i = 0; i < 8 && all_card; i++) {
		size_t v_free, v_total;
		printf("cudaSetDevice on device %d start, any key to continue\n", i);getchar();
		cudaSetDevice(i);
		printf("======cudaMalloc on device %d end, any key to continue\n", i);getchar();
		int* d_A;
		cudaMalloc(&d_A, size);
	    int* d_B;
	    cudaMalloc(&d_B, size);
	    int* d_C;
	    cudaMalloc(&d_C, size);
		printf("======cudaMemcpy on gpu:%d, any key to continue\n", i);getchar();

		// Copy vectors from host memory to device memory
		cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
		memset(h_C, 0x0, size);
		printf("======VecAdd run kernel on gpu:%d, any key to continue\n", i);getchar();

		// Invoke kernel
		int threadsPerBlock = 256;
		int blocksPerGrid =
		        (N + threadsPerBlock - 1) / threadsPerBlock;
//		VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
		VecAdd2<<<1, N>>>(d_A, d_B, d_C);

		printf("======cudaMemcpy run kernel on gpu:%d, any key to continue\n", i);getchar();
		// h_C contains the result in host memory
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		for (int j = 0; j < size / 4; j++) {
			if (h_C[j] != 0x33333333) {
				printf("NOT espect value, :%f\n", h_C[j]);
				all_card = 0;
				print_array(h_C, N);
				break;
			}
		}

//		cudaMemGetInfo(&v_free, &v_total);
		printf("======cudaFree run kernel on gpu:%d, any key to continue\n", i);getchar();
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);


	return 0;
}


int main_v4()
{
	size_t size = 0x21000000;
	
    int* h_A = (int*)malloc(size);
	memset(h_A, 0x11111111, size);
	int all_card = 1;
	
	for (int i = 0; i < 8 && all_card; i++) {
		printf("cudaSetDevice on device %d start, any key to continue\n", i);getchar();
		cudaSetDevice(i);
		printf("======cudaFree on device %d, any key to continue\n", i);getchar();
		cudaFree(0);
		printf("======cudaMalloc on device %d, any key to continue\n", i);getchar();
		int* d_A;
		cudaMalloc(&d_A, size);
		printf("======cudaMemcpy on gpu:%d, any key to continue\n", i);getchar();

		// Copy vectors from host memory to device memory
		cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

		printf("======cudaFree run kernel on gpu:%d, any key to continue\n", i);getchar();
		cudaFree(d_A);
	}

    // Free host memory
    free(h_A);

	return 0;
}

int main_v5()
{
	int N = 0x10000;
	int size = N * sizeof(int);
	
    int* h_A = (int*)malloc(size);
	memset(h_A, 0x11111111, size);
	int all_card = 1;
	
	for (int i = 0; i < 8 && all_card; i++) {
		printf("cudaSetDevice on device %d start, any key to continue\n", i);getchar();
		cudaSetDevice(i);
		printf("======cudaMalloc on device %d end, any key to continue\n", i);getchar();
		int* d_A;
		cudaMalloc(&d_A, size);
		printf("======cudaMemcpy on gpu:%d, any key to continue\n", i);getchar();

//		cudaPointerAttributes *cuda_p_attr = malloc(cudaPointerAttributes);
		cudaPointerAttributes cuda_p_attr;
		
		cudaPointerGetAttributes(&cuda_p_attr, d_A);
		printf("d_A:0x%p, from device:%d, host_p:0x%p, device_p:0x%p, type:%d\n",
//			d_A, cuda_p_attr->device, cuda_p_attr->hostPointer, cuda_p_attr->devicePointer, (int)(cuda_p_attr->type));
			d_A, cuda_p_attr.device, cuda_p_attr.hostPointer, cuda_p_attr.devicePointer, (int)(cuda_p_attr.type));


		// Copy vectors from host memory to device memory
		cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

		printf("======cudaFree run kernel on gpu:%d, any key to continue\n", i);getchar();
		cudaFree(d_A);
	}

    // Free host memory
    free(h_A);

	return 0;
}


int main(int argc, char const *argv[])
{
	main_v4();
    return 0;
}


