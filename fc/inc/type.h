#ifndef _TYPE_H_
#define _TYPE_H_

enum {W=0, H=1, C=2, N=3};

typedef struct _FcInfo {
    int ifmDim[4];
    int output_num;
} FcInfo;

#endif
