#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void selfAdd(float *dPtr, int num)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num)
        dPtr[i] *= 2;
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    int size = 2 * 1024 * 1024;
    float *hPtr = NULL;
    float *dPtr = NULL;
    int blocksPerGrid = (size / sizeof (float) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    hPtr = (float *)malloc(size);
    if (!hPtr)
    {
        printf("Failed to allocate host memory\n");
        return -1;
    }
    else
    {
        for (int i = 0; i < size / sizeof(float); i++)
            hPtr[i] = 1.0;
    }

    err = cudaMalloc((void **)&dPtr, size);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate cuda memory\n");
        return -1;
    }

    err = cudaMemcpy(dPtr, hPtr, size, cudaMemcpyHostToDevice);
    if (err  != cudaSuccess)
    {
        printf("Failed to copy host memory to device memory\n");
        return -1;
    }

    printf("Start launching kernel\n");
    selfAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dPtr, size / sizeof (float));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch cuda kernel\n");
        return -1;
    }
    printf("Exec kernel successfully\n");

    err = cudaMemcpy(hPtr, dPtr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy the result to host memory\n");
        return -1;
    }

    for (int i  = 0; i < size / sizeof(float); i++)
    {
        if (hPtr[i] != 1.0 * 2)
        {
            printf("Execution result is not correct!\n");
            return -1;
        }
    }

    printf("Exit successfully\n");
    cudaFree(dPtr);
    free(hPtr);
    return 0;
}
