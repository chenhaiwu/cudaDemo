#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

// argv[1]: malloc size;
// argv[2]: gpuidx
int main(int argc, char **argv)
{
    cudaError_t err = cudaSuccess;
    //size_t size = 0x10000000;//256M
    void *dPtr = NULL;

    if (argc < 3) printf("paramter error\n");
    
    size_t size = strtol(argv[1], NULL, 16);
    int gpuidx = atoi(argv[2]);
    printf("try to malloc %lx from GPU:%d\n", size, gpuidx);


//    printf("size_t len:%lu\n", sizeof(size_t)); //8 bytes
    cudaSetDevice(gpuidx);
    cudaFree(0);
    err = cudaMalloc(&dPtr, size);

    if (err != cudaSuccess)
    {
        printf("Failed to allocate cuda memory!\n");
        return -1;
    }

    printf("Allocated cuda 1G memory successfully, ptr=0x%016llx\n", dPtr);
    sleep(200000);

    return 0;
}
