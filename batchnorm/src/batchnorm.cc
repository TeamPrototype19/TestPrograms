#include "batchnorm.h"

#include <iostream>

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
) {

    //int dbg_out_cnt = 0;

    int N = rinfo.ifmDim[0];
    int C = rinfo.ifmDim[1];
    int H = rinfo.ifmDim[2];
    int W = rinfo.ifmDim[3];
    float eps = rinfo.eps;

    float s = (*scale == 0) ? 0 : 1/(*scale);

    /*
    float *_o = output;
    std::cout << "mean  = " << *mean << std::endl;
    std::cout << "var   = " << *var  << std::endl;
    std::cout << "sclae = " << s << std::endl;
    std::cout << "eps   = " << eps << std::endl;
    std::cout << "in[0] = " << *input << std::endl;
    */

    for(int n = 0 ; n < N ; n++) {
        for(int c = 0 ; c < C; c++) {
            float m = *mean++ * s;
            float v = sqrt((*var++ * s) + eps);
            for(int i = 0 ; i < W*H; i++) {
                *output++ = (*input++ - m) / v;
            }
        }
    }

    //std::cout << "out[0]= " << *_o << std::endl;

    return;
}
