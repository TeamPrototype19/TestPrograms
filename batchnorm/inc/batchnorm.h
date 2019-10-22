#include "type.h"
#include <cmath>

void test_kernel_batchnorm(
    // output
    float *output,
    // inputs
    float *input,
    float *mean,
    float *var,
    float *scale,
    // operation information
    BatchNormInfo rinfo
);
