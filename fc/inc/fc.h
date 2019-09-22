#include "type.h"

void test_kernel_fc(
    // output
    float *output,
    // inputs
    float *input,
    float *weight,
    float *bias,
    // operation information
    FcInfo finfo
);
