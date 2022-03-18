#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

int main(void)
{

    cudaError_t err = cudaSuccess;
    int size = 10*1024*1024;
    void *dPtr = NULL;


    for (int i = 0; i < 2; i ++) {
        cudaSetDevice(i);
        printf("Try to allocate cuda memory 10M\n");
        err = cudaMalloc(&dPtr, size);

        if (err != cudaSuccess)
        {
            printf("Failed to allocate cuda memory: err:%d, %s!\n", (int)err, cudaGetErrorString(err));
            return -1;
        }

        printf("Allocated cuda memory successfully\n");

        err = cudaMemset(dPtr, 0x0, size);

        cudaFree(dPtr);
    }

    sleep(20000);

    return 0;
}
