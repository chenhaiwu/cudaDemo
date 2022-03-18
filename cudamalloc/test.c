#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

// argv[1]: malloc size;
// argv[2]: gpuidx
int main(int argc, char **argv)
{
    unsigned long v9 = 4;
    __int128_t v11 = 0;
    unsigned long v10 = 80 * ((unsigned __int64)(0xCCCCCCCCCCCCCCCDLL * (unsigned __int128)(unsigned __int64)v9 >> 64) >> 6);
    printf("v9=%llx, v10=%llx\n", v9, v10, v11);

    return 0;
}
