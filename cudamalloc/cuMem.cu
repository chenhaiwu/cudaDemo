#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{

    cudaError_t err = cudaSuccess;
    int size = 1024*1024*1024;
    void *dPtr = NULL;

    err = cudaMalloc(&dPtr, size);

    if (err != cudaSuccess)
    {
        printf("Failed to allocate cuda memory!\n");
        return -1;
    }

    printf("Allocated cuda memory successfully\n");

    err = cudaMemset(dPtr, 0x0, size);

//    cudaFree(dPtr);

    //cudaFree(0);
    return 0;
}
