#ifndef _TYPE_H_
#define _TYPE_H_

enum {W=0, H=1, C=2, N=3};

typedef struct _PoolInfo {
    int ifmDim[4];
    int kernel_size_w;
    int kernel_size_h;
    int stride_size_w;
    int stride_size_h;
    int pad_size_w;
    int pad_size_h;
    int pool_type;
    int global_pooling;
} PoolInfo;

#endif
