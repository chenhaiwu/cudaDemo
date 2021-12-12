# coding=utf8
import sys
import re
import os
import time

opt = ""
val = 2
base_addr = 0


def multiDev_compile(nums=8):
    for i in range(nums):
        file_name = "cuMemMalloc.cu"
        target_name = "cuMemMalloc_" + str(i)
        cmd = "nvcc -o " + target_name + " " + file_name
        print(cmd)
        os.system(cmd)
    return


def mulDev_execute(nums=8):
    for i in range(nums):
        target_name = "cuMemMalloc_" + str(i)
        alloc_size = 0x10000000 * (i + 1) + 0x1000000
        cmd = "./" + target_name + " " + hex(alloc_size) + " " + str(i) + " &"
        # cmd = "./" + target_name + " 0x1000000 &"
        print(cmd)
        os.system(cmd)
        time.sleep(3)
    return


if __name__ == '__main__':
    # base_addr = int(sys.argv[1], 16)
    opt = sys.argv[1]
    pgpunums = int(sys.argv[2])
    if opt == "comp":
        multiDev_compile(pgpunums)
    elif opt == "exe":
        mulDev_execute(pgpunums)
    else:
        print("Not support cmd")
