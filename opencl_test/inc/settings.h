//kernel header/source file path
#define CL_INCLUDE_FILE "./inc/settings.h"
#define CL_KERNEL_FILE  "./kernel/kernels.cl"

//OpenCL settings
#define MAX_NUM_DEVICES 16
#define MAX_DEVICE_NAME 1024
#define CURRENT_DEVICE 0
#define MAX_SOURCE_SIZE (0x100000)

//debug dump out data
//#define debug

//gemm setting
#define KERNEL 1

#define TS 10
#define TS_w 1
#define TS_h 24

#define WPT 8
#define RTS (TS/WPT)

#define WIDTH 4
#define TSDK 16


#define MIN(a,b) ((a)>(b))? (b):(a)
#define MAX(a,b) ((a)>(b))? (a):(b)
#define CEIL_DIV (x,y) (((x)+(y)-1)/(y))
#define MOD2(x,y) ((x)%(y))
#define DIV2(x,y) ((x)/(y))




