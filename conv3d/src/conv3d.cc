#include "conv3d.h"

void test_kernel_conv3d(
    // output
    float *output,
    // inputs
    float *input,
    float *weight,
    float *bias,
    // operation information
    ConvInfo cinfo
) {

    int N = cinfo.ifmDim[N];
    int C = cinfo.ifmDim[C];
    int H = cinfo.ifmDim[H];
    int W = cinfo.ifmDim[W];
    int O = cinfo.output_num;
    int KW = cinfo.kernel_size_w;
    int KH = cinfo.kernel_size_h;
    int SW = cinfo.stride_size_w;
    int SH = cinfo.stride_size_h;
    int PW = cinfo.pad_size_w;
    int PH = cinfo.pad_size_h;

    for(int n = 0 ; n < N ; n++) {
    for(int o = 0 ; o < O ; o++) {
        float sum = 0;

        /* IFM loop (3D)
         */
        for(int c = 0 ; c < C ; c++) {
            for(int h = 0 ; h < H ; h++) {
                for(int w = 0 ; w < W ; w++) {

                    /* Kernel loop
                     */
                    if( (w+KW) < W && (h+KH) < H ) {
                        for(int kh = 0 ; kh < KH ; kh++) {
                            float *i = input + (W*(h+kh)) + (W*H*c);
                            float *w = weight + (KW*kh) + (KW*KH*c);
                            for(int kw = 0 ; kw < KW ; kw++) {
                                sum += (*i++) * (*w++);
                            }
                        }
                    }

                }
            }
        }
        *output++ = sum;

    }
    }
}
