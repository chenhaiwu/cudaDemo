#include <builtin_types.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#include <cstring>
#include <iostream>
#include <string>


CUdevice cuDevice_0;
CUdevice cuDevice_1;

CUcontext cuContext_0;
CUcontext cuContext_1;

inline void checkCudaErrors(CUresult result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString((cudaError_t)result));
		exit(EXIT_FAILURE);
	}
}

void printhex(void *p, int len)
{
	int i = 0;
	unsigned long long *pp = (unsigned long long *)p;
	printf("==============haiwu dump addr:0x%016x===================\n", pp);
	for (i = 0; i < len / 8; i++) {
		if (i % 2 == 0) {
			printf("0x%08x\t\t", i * 8);
		}
		printf("%016x  ", (unsigned int)(*(pp + i)));
		if (i % 2 == 1) {
			printf("\n");
		}
	}
	printf("\n");
}


static CUresult initCUDA(int dev_id) {
	CUfunction cuFunction = 0;
	CUresult status;
	int major = 0, minor = 0;
	char deviceName[100];
	std::string module_path, ptx_source;
	CUdevice cuDevice = dev_id == 0 ? cuDevice_0 : cuDevice_1;
	CUcontext cuContext = dev_id == 0 ? cuContext_0 : cuContext_1;
	int ctx_size = sizeof(cuContext);


	checkCudaErrors(cuDeviceGet(&cuDevice, dev_id));
	cuDeviceGetName(deviceName, 100, cuDevice);
	printf("> Using CUDA Device [%d]: %s\n", dev_id, deviceName);

	// get compute capabilities and the devicename
	checkCudaErrors(cuDeviceGetAttribute(
	  &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
	checkCudaErrors(cuDeviceGetAttribute(
	  &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
	checkCudaErrors(cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice));
	printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

	status = cuCtxCreate(&cuContext, 0, cuDevice);
	printf("haiwu================2.5 cuCtxCreate done\n");
	printf("ctx size: %d, print conext 0x2900:\n", ctx_size);
//	printf("sizeof(CUcontext):%d, sizeof(CUcontext):%d\n", p_ctx_size, ctx_size);
	printhex(cuContext, 0x2900);

	return CUDA_SUCCESS;
Error:
  cuCtxDestroy(cuContext);
  return status;
}


int main()
{
	printf("create context on device 0 start\n");
	CUresult err = cuInit(0);
	if (CUDA_SUCCESS != err) {
		printf("cuInit fail\n");
	}
	initCUDA(0);
	printf("create context on device 0 end\n");

	sleep(5);

	printf("create context on device 1 start\n");
	initCUDA(1);
	printf("create context on device 1 done\n");
	sleep(5);

	return 0;
}

/*
int main(int argc, char const *argv[])
{
    if (InitCUDA()) {
        printf("CUDA initialized.\n");
    }
 
    return 0;
}
*/

