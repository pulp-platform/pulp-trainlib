#include "test_utils.h"


void set_array_fp32(float *array, size_t size, float value)
{
    for (int i = 0; i < size; i++) {
        array[i] = value;
    }
}

