#include "softmax.h"

#include <iostream>
#include <cmath>

void test_kernel_softmax(
    // output
    float *output,
    // inputs
    float *input,
    // operation information
    SoftmaxInfo sinfo
) {

    //int dbg_out_cnt = 0;

    int N = sinfo.ifmDim[0];
    int C = sinfo.ifmDim[1];
    int H = sinfo.ifmDim[2];
    int W = sinfo.ifmDim[3];

    int size = C*H*W;

    for(int n = 0 ; n < N; n++) {
        float sum = 0;
        float *op = output;
        /* Calculate sum
         */
        for(int i = 0 ; i < size ; i++) {
            *op = exp(*input++);
            sum += *op++;
        }
        for(int i = 0 ; i < size ; i++) {
            *output++ /= sum;
        }
    }

    return;
}
