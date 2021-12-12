#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

int main(void)
{

    cudaError_t err = cudaSuccess;
    size_t size = 0xcc000000;//3G
    void *dPtr = NULL;

//    printf("size_t len:%lu\n", sizeof(size_t)); //8 bytes
    cudaFree(0);
    err = cudaMalloc(&dPtr, size);

    if (err != cudaSuccess)
    {
        printf("Failed to allocate cuda memory!\n");
        return -1;
    }

    printf("Allocated cuda 1G memory successfully, ptr=0x%016llx\n", dPtr);
    sleep(20);

    return 0;
}
