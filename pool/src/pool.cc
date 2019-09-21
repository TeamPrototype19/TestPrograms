#include "pool.h"

#include <iostream>

void test_kernel_pool(
    // output
    float *output,
    // inputs
    float *input,
    // operation information
    PoolInfo pinfo
) {

    //int dbg_out_cnt = 0;

    int N = pinfo.ifmDim[0];
    int C = pinfo.ifmDim[1];
    int H = pinfo.ifmDim[2];
    int W = pinfo.ifmDim[3];
    int KW = pinfo.kernel_size_w;
    int KH = pinfo.kernel_size_h;
    int SW = pinfo.stride_size_w;
    int SH = pinfo.stride_size_h;
    int PW = pinfo.pad_size_w;
    int PH = pinfo.pad_size_h;
    int TP = pinfo.pool_type;
    int GP = pinfo.global_pooling;

    for(int n = 0 ; n < N ; n++) {
    for(int c = 0 ; c < C ; c++) {
        for(int h = -PH ; h < (H+PH) ; h += SH) {
        for(int w = -PW ; w < (W+PW) ; w += SW) {

            /* Kernel loop
             */
            if( (w+KW) <= (W+PW) && (h+KH) <= (H+PH) ) {
                float elem = -INFINITY;
                int   elem_cnt = 0;

                //std::cout << "[" << dbg_out_cnt++ << "]\t";

                for(int kh = 0 ; kh < KH ; kh++) {
                    if( (h+kh) >= 0 && (h+kh) < H ) {
                        float *inp = input + (W*(h+kh)) + (W*H*c) + w + (C*H*W*n);
                        for(int kw = 0 ; kw < KW ; kw++) {
                            if( (w+kw) >= 0 && (w+kw) < W ) {
                                if( TP == 0 ) { // MAX_POOL
                                    elem = (elem < (*inp)) ? (*inp) : elem;
                                    //std::cout << *inp << "\t";
                                }
                                else if( TP == 1 ) { // AVG_POOL
                                    elem += (*inp);
                                    elem_cnt++;
                                }
                            }
                            inp++;
                        }
                    }
                }

                if( TP == 1 )
                    elem /= (float)elem_cnt;

                *output++ = elem;

                //std::cout << "Final = " << elem << std::endl;

            }
        }
        }
    }
    }
}
