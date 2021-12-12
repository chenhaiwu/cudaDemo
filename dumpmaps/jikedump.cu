#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdlib.h>

static void dump(void *start_va, int size)
{
	int i;
	long v;
	for (i = 0; i < size / 8; i++) {
		v = *((long *)start_va + i);
		printf("%x:	%0.16lx\n", i, v);
	}
}


int main(void)
{

	cudaError_t err = cudaSuccess;
	size_t size =  0x3A00;
	void *dPtr = NULL;

	printf("size = %llx\n", size);
	err = cudaMalloc(&dPtr, size);

	if (err != cudaSuccess)
	{
		printf("Failed to allocate cuda memory!\n");
		exit(1);
	}

	printf("Allocated cuda memory successfully\n");

	long start_va = 0, end;
	char buf[PATH_MAX];
	FILE *fp = fopen("/proc/self/maps", "r");
	while (fgets(buf, sizeof(buf), fp) != NULL) {
		if (strstr(buf, "/dev/nvidia0")) {
			printf("buf: %s\n", buf);
			break;
		}
	}

	char *p = strchr(buf, ' ');
	if (p)
		*p = '\0';
	sscanf(buf, "%lx-%lx", &start_va, &end);

	dump((void *)start_va, 4096);



	cudaFree(dPtr);
	return 0;
}
