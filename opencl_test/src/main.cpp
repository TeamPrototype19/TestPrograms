#include <iostream>
#include <fstream>
#include <sstream>
#include <malloc.h>

#include "ssg_kernel.h"
#include <time.h>
#include <sys/time.h>

int main()
{
    //kernel input matirx
    int weight_h = 25;
    int weight_w = 30;
    int img_n =100;
    int img_c = 1;
    int img_h = 28;
    int img_w = 28;
    int f_n =30;
    int f_c =1;
    int f_h=5;
    int f_w=5;

    float * img_in = new float[img_n*img_c*img_h*img_w];
    float * weight_in = new float[weight_h*weight_w];
    float * b_in = new float[weight_w];
    float * col_out = new float[1440000];
    
    FILE * fp_tmp;
    fp_tmp = fopen("in/x.txt","r");
    float tmp_val =0;
    for(int iter=0;iter<img_n*img_c*img_h*img_w;iter++){
        fscanf(fp_tmp,"%f\n",&tmp_val);
        img_in[iter]=tmp_val;
    }
    fclose(fp_tmp);

    fp_tmp = fopen("in/col_w.txt","r");
    tmp_val =0;
    for(int iter=0;iter<weight_w*weight_h;iter++){
        fscanf(fp_tmp,"%f\n",&tmp_val);
        weight_in[iter]=tmp_val;
    }
    fclose(fp_tmp);

    fp_tmp = fopen("in/b.txt","r");
    tmp_val =0;
    for(int iter=0;iter<weight_w;iter++){
        fscanf(fp_tmp,"%f\n",&tmp_val);
        b_in[iter]=tmp_val;
    }
    fclose(fp_tmp);
    
    //create kernel class & get device resource
    ssg_kernel mkernel(img_in,img_w,img_h,img_c,img_n,weight_in,weight_w,weight_h,b_in,col_out);
    mkernel.get_cl_info();
    mkernel.init_device();
    mkernel.compile_program();

    struct timeval startTime, endTime, gepTime;
    
    //im2col
    mkernel.Im2col.im2col_init(30,5,1,0,0);
    
    //gettimeofday( &startTime, NULL );
      mkernel.Im2col.im2col_cpu_();
    //gettimeofday( &endTime, NULL );
    //gepTime.tv_sec = endTime.tv_sec - startTime.tv_sec;
    // gepTime.tv_usec = endTime.tv_usec - startTime.tv_usec;
    // printf("im2col cpu time [%02d.%02d] second\n", gepTime.tv_sec, gepTime.tv_usec);
    //if ( gepTime.tv_usec < 0 )	{
	//gepTime.tv_sec = gepTime.tv_sec - 1;
	//gepTime.tv_usec = gepTime.tv_usec + 1000000;
	//}
    //printf("gemm cpu time [%02d.%02d] second\n", gepTime.tv_sec, gepTime.tv_usec);
   
    
    //gettimeofday( &startTime, NULL );
    mkernel.Im2col.im2col_gpu();
    //gettimeofday( &endTime, NULL );
    //gepTime.tv_sec = endTime.tv_sec - startTime.tv_sec;
    //gepTime.tv_usec = endTime.tv_usec - startTime.tv_usec;
    //printf("im2col gpu  time [%02d.%02d] second\n", gepTime.tv_sec, gepTime.tv_usec);
    //if ( gepTime.tv_usec < 0 )	{
	//gepTime.tv_sec = gepTime.tv_sec - 1;
	//gepTime.tv_usec = gepTime.tv_usec + 1000000;
	//}
    //printf("gemm cpu time [%02d.%02d] second\n", gepTime.tv_sec, gepTime.tv_usec);
    
    
    //gemm
    mkernel.Gemm.gemm_init();
   
    gettimeofday( &startTime, NULL );
    mkernel.Gemm.gemm_cpu();
    gettimeofday( &endTime, NULL );
    gepTime.tv_sec = endTime.tv_sec - startTime.tv_sec;
    gepTime.tv_usec = endTime.tv_usec - startTime.tv_usec;
    if ( gepTime.tv_usec < 0 )	{
	gepTime.tv_sec = gepTime.tv_sec - 1;
	gepTime.tv_usec = gepTime.tv_usec + 1000000;
	}
    printf("gemm cpu time [%02d.%02d] second\n", gepTime.tv_sec, gepTime.tv_usec);
    
    
    gettimeofday( &startTime, NULL );
    mkernel.Gemm.gemm_gpu();
    gettimeofday( &endTime, NULL );
    gepTime.tv_sec = endTime.tv_sec - startTime.tv_sec;
    gepTime.tv_usec = endTime.tv_usec - startTime.tv_usec;
    if ( gepTime.tv_usec < 0 )	{
	gepTime.tv_sec = gepTime.tv_sec - 1;
	gepTime.tv_usec = gepTime.tv_usec + 1000000;
	}
    printf("gemm  gpu  time [%02d.%02d] second\n", gepTime.tv_sec, gepTime.tv_usec);
    
    //im2col
    mkernel.Im2col.im2col_init(1,2,2,0,1);
    mkernel.Im2col.im2col_cpu_();
    mkernel.Im2col.im2col_gpu();

    //maxpool
    mkernel.Maxpool.maxpool_init();
    mkernel.Maxpool.maxpool_cpu();
    mkernel.Maxpool.maxpool_gpu();
    

    delete [] b_in;
    delete [] img_in;
    delete [] weight_in;
    delete [] col_out;

				
    return 0;
}

