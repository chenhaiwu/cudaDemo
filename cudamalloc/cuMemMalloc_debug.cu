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


//    printf("size_t len:%lu\n", sizeof(size_t)); //8 bytes
    cudaSetDevice(gpuidx);
    cudaFree(0);
    for (int t = 0; t < 3; t++) {
        printf("try to malloc %lx from GPU:%d\n", size, gpuidx);getchar();
        err = cudaMalloc(&dPtr, size);
        if (err != cudaSuccess)
        {
            printf("Failed to allocate cuda memory!\n");
            break;
        }
        printf("try to free this cudamem:%16lx\n", (unsigned long)dPtr);getchar();
        err = cudaFree(dPtr);
        if (err != cudaSuccess)
        {
            printf("Failed to free cuda memory!\n");
            break;
        }
    }

    sleep(200000);

    return 0;
}
