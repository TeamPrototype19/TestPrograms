#include "relu.h"

#include <iostream>

void test_kernel_relu(
    // output
    float *output,
    // inputs
    float *input,
    // operation information
    ReluInfo rinfo
) {

    //int dbg_out_cnt = 0;

    int N = rinfo.ifmDim[0];
    int C = rinfo.ifmDim[1];
    int H = rinfo.ifmDim[2];
    int W = rinfo.ifmDim[3];
    int TR = rinfo.relu_type;

    int size = N*C*H*W;

    for(int n = 0 ; n < size ; n++) {
        /* Kernel loop
         */
        if( *input < 0 )
            *output++ = 0;
        else
            *output++ = *input;
        input++;
    }

    return;
}
