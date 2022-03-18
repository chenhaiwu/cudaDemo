#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

// argv[1]: malloc size;
// argv[2]: gpuidx
int main(int argc, char **argv)
{
    cudaError_t err = cudaSuccess;
    void *dPtr = NULL;

    if (argc < 3) printf("paramter error\n");
    
    size_t size = strtol(argv[1], NULL, 16);
    int gpuidx = atoi(argv[2]);
    printf("try to malloc %lx from GPU:%d, init first\n", size, gpuidx);

//    printf("size_t len:%lu\n", sizeof(size_t)); //8 bytes
    cudaSetDevice(gpuidx);
    cudaFree(0);

    printf("try to get meminfo after any key\n");getchar();
    size_t freesize, totalsize;
    err = cudaMemGetInfo(&freesize, &totalsize);
    if (err != cudaSuccess) {
        printf("Failed to get memory info:%d, %s\n", err, cudaGetErrorString(err));
        return -1;
    }
    printf("The GPU:%d have free size:%lx, total size:%lx\n", gpuidx, freesize, totalsize);

    printf("try to malloc %lx from GPU:%d\n", size, gpuidx);getchar();
    err = cudaMalloc(&dPtr, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate cuda memory: err%d, %s!\n", err, cudaGetErrorString(err));
        return -1;
    }

    printf("try to get meminfo2 after any key\n");getchar();
    err = cudaMemGetInfo(&freesize, &totalsize);
    printf("The GPU:%d have free size:%lx, total size:%lx\n", gpuidx, freesize, totalsize);
    if (err != cudaSuccess) {
        printf("Failed to get memory info!\n");
        return -1;
    }
    sleep(200000);

}
