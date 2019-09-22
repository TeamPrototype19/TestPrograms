#include "fc.h"

void test_kernel_fc(
    // output
    float *output,
    // inputs
    float *input,
    float *weight,
    float *bias,
    // operation information
    FcInfo finfo
) {

    int N = finfo.ifmDim[0];
    int C = finfo.ifmDim[1];
    int H = finfo.ifmDim[2];
    int W = finfo.ifmDim[3];
    int O = finfo.output_num;

    int ifm_size = C*H*W;


    float *wg = weight;
    float *bp = bias;
    for(int n = 0 ; n < N ; n++) {
        for(int o = 0 ; o < O ; o++) {
            float *in = input + ifm_size * n;
            float sum = 0;
            for(int i = 0 ; i < ifm_size ; i++) {
                sum += (*in++) * (*wg++);
            }
            *output++ = sum + (*bias++);
        }
    }

    return;
}
