#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <nvml.h>

#define JDEBUG(fmt, ...) do { printf("[%s-%d] "fmt, __func__, __LINE__, ##__VA_ARGS__); } while (0)

int main()
{
	int i, device_count;
	nvmlReturn_t result;
	nvmlDevice_t device;
	nvmlUnit_t unit;
	nvmlPciInfo_t pci;
	char name[NVML_DEVICE_NAME_BUFFER_SIZE];

	JDEBUG("press anykey to call nvmlInit()..."); getchar();
	result = nvmlInit();
	if (NVML_SUCCESS != result) {
		JDEBUG("Failed to initialize NVML: %s\n", nvmlErrorString(result));
		return 1;
	}

	JDEBUG("press anykey to call nvmlDeviceGetCount()..."); getchar();
	result = nvmlDeviceGetCount(&device_count);
	if (NVML_SUCCESS != result)
		JDEBUG("Failed to query device count: %s\n", nvmlErrorString(result));
	else
		JDEBUG("device count: %d\n", device_count);


	JDEBUG("Listing devices:\n");
	for (i = 0; i < device_count; i++) {
		JDEBUG("press anykey to call nvmlDeviceGetHandleByIndex()..."); getchar();
		result = nvmlDeviceGetHandleByIndex(i, &device);
		if (NVML_SUCCESS != result)
			JDEBUG("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));

		JDEBUG("press anykey to call nvmlDeviceGetName()..."); getchar();
		result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
		if (NVML_SUCCESS != result)
			JDEBUG("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));

		JDEBUG("press anykey to call nvmlDeviceGetPciInfo()..."); getchar();
		result = nvmlDeviceGetPciInfo(device, &pci);
		if (NVML_SUCCESS != result)
			JDEBUG("Failed to get pci info for device %i: %s\n", i, nvmlErrorString(result));
		else {
			JDEBUG("%d. %s [%s]\n", i, name, pci.busId);
			JDEBUG("PCI Device: %x, PCI Subsystem: %x\n", pci.pciDeviceId, pci.pciSubSystemId);
			JDEBUG("sizeof(pci): %d\n", sizeof(pci));
		}

		JDEBUG("press anykey to call nvmlDeviceGetUUID()..."); getchar();
		char uuid[NAME_MAX] = "";
		result = nvmlDeviceGetUUID(device, uuid, sizeof(uuid));
		if (NVML_SUCCESS != result)
			JDEBUG("Failed to get UUID info for device %i: %s\n", i, nvmlErrorString(result));
		else
			JDEBUG("GPU UUID: %s\n", uuid);
	}

	JDEBUG("press anykey to call nvmlDeviceGetUUID()..."); getchar();
	result = nvmlShutdown();
	if (NVML_SUCCESS != result)
		JDEBUG("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
	else
		JDEBUG("nvmlShutdown: done\n");
	return 0;
}
