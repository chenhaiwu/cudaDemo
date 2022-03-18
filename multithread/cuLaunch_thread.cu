#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <pthread.h>

#define THREADS_PER_BLOCK 256

__global__ void selfAdd(float *dPtr, int num)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num)
        dPtr[i] *= 2;
}


#define PRINT_ERROR \
	do { \
		fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
		__LINE__, __FILE__, errno, strerror(errno)); exit(1); \
	} while(0)


int qgpu_mmap_test(int argc, char **argv) {
	int fd;
	void *map_base, *virt_addr;
	uint64_t read_result, writeval, prev_read_result = 0;
	char *filename;
	off_t target, target_base;
	int access_type = 'w';
	int items_count = 1;
	int verbose = 0;
	int read_result_dupped = 0;
	int type_width;
	int i;
	int map_size = 4096UL;

	if(argc < 3) {
		// pcimem /sys/bus/pci/devices/0001\:00\:07.0/resource0 0x100 w 0x00
		// argv[0]  [1]                                         [2]   [3] [4]
		fprintf(stderr, "\nUsage:\t%s { sysfile } { offset } [ type*count [ data ] ]\n"
			"\tsys file: sysfs file for the pci resource to act on\n"
			"\toffset  : offset into pci memory region to act upon\n"
			"\ttype    : access operation type : [b]yte, [h]alfword, [w]ord, [d]ouble-word\n"
			"\t*count  : number of items to read:  w*100 will dump 100 words\n"
			"\tdata    : data to be written\n\n",
			argv[0]);
		exit(1);
	}
	filename = argv[1];
	target = strtoul(argv[2], 0, 0);

	if(argc > 3) {
		access_type = tolower(argv[3][0]);
		if (argv[3][1] == '*')
			items_count = strtoul(argv[3]+2, 0, 0);
	}

        switch(access_type) {
		case 'b':
			type_width = 1;
			break;
		case 'h':
			type_width = 2;
			break;
		case 'w':
			type_width = 4;
			break;
                case 'd':
			type_width = 8;
			break;
		default:
			fprintf(stderr, "Illegal data type '%c'.\n", access_type);
			exit(2);
	}

printf("filename:%s\n", filename);
    if((fd = open(filename, O_RDWR | O_SYNC)) == -1) PRINT_ERROR;
    printf("%s opened.\n", filename);
    printf("Target offset is 0x%x, page size is %ld\n", (int) target, sysconf(_SC_PAGE_SIZE));
    fflush(stdout);

    target_base = target & ~(sysconf(_SC_PAGE_SIZE)-1);
    if (target + items_count*type_width - target_base > map_size)
	map_size = target + items_count*type_width - target_base;

    /* Map one page */
    printf("mmap(%d, %d, 0x%x, 0x%x, %d, 0x%x)\n", 0, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (int) target);

    map_base = mmap(0, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, target_base);
    if(map_base == (void *) -1) {
        return 0;
    }; 
    // if(map_base == (void *) -1) PRINT_ERROR;
    printf("PCI Memory mapped to address 0x%08lx.\n", (unsigned long) map_base);
    fflush(stdout);

    for (i = 0; i < items_count; i++) {

        virt_addr = map_base + target + i*type_width - target_base;
        switch(access_type) {
		case 'b':
			read_result = *((uint8_t *) virt_addr);
			break;
		case 'h':
			read_result = *((uint16_t *) virt_addr);
			break;
		case 'w':
			read_result = *((uint32_t *) virt_addr);
			break;
                case 'd':
			read_result = *((uint64_t *) virt_addr);
			break;
	}

    	if (verbose)
            printf("Value at offset 0x%X (%p): 0x%0*lX\n", (int) target + i*type_width, virt_addr, type_width*2, read_result);
        else {
	    if (read_result != prev_read_result || i == 0) {
                printf("0x%04X: 0x%0*lX\n", (int)(target + i*type_width), type_width*2, read_result);
                read_result_dupped = 0;
            } else {
                if (!read_result_dupped)
                    printf("...\n");
                read_result_dupped = 1;
            }
        }
	
	prev_read_result = read_result;

    }

    fflush(stdout);

	if(argc > 4) {
		writeval = strtoull(argv[4], NULL, 0);
		switch(access_type) {
			case 'b':
				*((uint8_t *) virt_addr) = writeval;
				read_result = *((uint8_t *) virt_addr);
				break;
			case 'h':
				*((uint16_t *) virt_addr) = writeval;
				read_result = *((uint16_t *) virt_addr);
				break;
			case 'w':
				*((uint32_t *) virt_addr) = writeval;
				read_result = *((uint32_t *) virt_addr);
				break;
			case 'd':
				*((uint64_t *) virt_addr) = writeval;
				read_result = *((uint64_t *) virt_addr);
				break;
		}
		printf("Written 0x%0*lX; readback 0x%*lX\n", type_width,
		       writeval, type_width, read_result);
		fflush(stdout);
	}

	if(munmap(map_base, map_size) == -1) PRINT_ERROR;
    close(fd);
    return 0;
}

char *s_pcimem = "pcimem";
char *s_dev = "/dev/nvidia0";
char *s_offset = "1024";
char *s_type = "w";
char *s_argv[4];

void *qgpu_mmap_thread(void *arg)
{
    int i;

    printf("qgpu_mmap_thread start\n");
    s_argv[0] = s_pcimem;
    s_argv[1] = s_dev;
    s_argv[2] = s_offset;
    s_argv[3] = s_type;

    for (i = 0; i < 10000; i++) {
        qgpu_mmap_test(3, s_argv);
        sleep(1);
    }

    printf("qgpu_mmap_thread done\n");
	return NULL;
}

void *qgpu_calc_thread(void *arg)
{
    cudaError_t err = cudaSuccess;
    int size = 20 * 1024 * 1024;
    float *hPtr = NULL;
    float *dPtr = NULL;
    int blocksPerGrid = (size / sizeof (float) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int i;

    printf("qgpu_calc_thread doing\n");

    hPtr = (float *)malloc(size);
    if (!hPtr)
    {
        printf("Failed to allocate host memory\n");
        return NULL;
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
        return NULL;
    }

    err = cudaMemcpy(dPtr, hPtr, size, cudaMemcpyHostToDevice);
    if (err  != cudaSuccess)
    {
        printf("Failed to copy host memory to device memory\n");
        return NULL;
    }

    printf("Start launching kernel\n");
    for (i = 0; i < 1000000; i++)
        selfAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dPtr, size / sizeof (float));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch cuda kernel\n");
        return NULL;
    }
    printf("Exec kernel successfully\n");

    err = cudaMemcpy(hPtr, dPtr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy the result to host memory\n");
        return NULL;
    }

    for (int i  = 0; i < size / sizeof(float); i++)
    {
        if (hPtr[i] != 1.0 * 2)
        {
            printf("Execution result is not correct!\n");
            return NULL ;
        }
    }

    printf("Exit successfully\n");
    cudaFree(dPtr);
    free(hPtr);
    return 0;
}

int g_pgpuid[2] = {0, 1};
int main(void)
{
    int ret;
    pthread_t tid[2];

    ret = pthread_create(&tid[0],NULL, qgpu_mmap_thread, (void*)&g_pgpuid[0]);
    if(ret != 0)
    {
        perror("pthread_create");
        return -1;
    }

    ret = pthread_create(&tid[1],NULL, qgpu_calc_thread, (void*)&g_pgpuid[1]);
    if(ret != 0)
    {
        perror("pthread_create");
        return -1;
    }

    printf("main thread doing\n");
    

	pthread_join(tid[0],NULL);
	pthread_join(tid[1],NULL);
    return 0;
}
