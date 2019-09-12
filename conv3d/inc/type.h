#ifndef _TYPE_H_
#define _TYPE_H_

typedef struct _ConvInfo {
    enum {W=0, H=1, C=2, N=3};
    int ifmDim[4];
    int kernel_size_w;
    int kernel_size_h;
    int stride_size_w;
    int stride_size_h;
    int pad_size_w;
    int pad_size_h;
    int output_num;
    int group_num;
} ConvInfo;

#endif
