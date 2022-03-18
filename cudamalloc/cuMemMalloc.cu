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
    void *dPtr2 = NULL;

    if (argc < 3) {
        printf("paramter error\n");
        return -1;
    }
    
    size_t size = strtol(argv[1], NULL, 16);
    int gpuidx = atoi(argv[2]);
    printf("try to malloc %lx from GPU:%d\n", size, gpuidx);


//    printf("size_t len:%lu\n", sizeof(size_t)); //8 bytes
    printf("try to init after any key\n");getchar();
    cudaSetDevice(gpuidx);
    printf("try to init gpu:%d after any key\n", gpuidx);getchar();
    cudaFree(0);
    printf("try to malloc size:%lx, on gpu:%d after any key\n", size, gpuidx);getchar();
    err = cudaMalloc(&dPtr, size);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate cuda memory: err:%d, %s!\n", (int)err, cudaGetErrorString(err));
        return -1;
    }
    size *= 2;
    printf("try to malloc 2 size:%lx, on gpu:%d after any key\n", size, gpuidx);getchar();
    err = cudaMalloc(&dPtr2, size);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate cuda memory: err:%d, %s!\n", (int)err, cudaGetErrorString(err));
        return -1;
    }
    cudaFree(dPtr);
    cudaFree(dPtr2);
    //sleep(200000);

    return 0;
}
