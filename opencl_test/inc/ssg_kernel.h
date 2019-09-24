#include <stdio.h>
#include <CL/cl.h>
#include "OpenCLInfo.h"
#include <stdlib.h>
#include <string.h>

#include "settings.h"
#include <alloca.h>
 

class ssg_kernel{
    public:

        ////////////////////////////////////////
        //im2col
        ///////////////////////////////////////
        class im2col{
            public:
                im2col(ssg_kernel &x):parent(x){
                }

                ~im2col(){
                }

                void im2col_init(int f_n_, int f_h_, int stride_,int pad_,int mode);
                void im2col_gpu();
                void im2col_cpu_();
                float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad);

                void im2col_cpu(float* data_im, int channels,  int height,  int width,
                                   int ksize,  int stride, int pad, float* data_col);
        

                //opencl: kernel, in_buf, out_buf
                cl_kernel   kernel_im2col;
                cl_int err;

                //in/out
                float * im2col_in;
                float * im2col_out;

                //parent class
                ssg_kernel & parent;

            private:
                int img_n,img_c,img_h,img_w;
                int f_n,f_c,f_h,f_w;
                int stride, pad;

                //out param
                int col_n, col_c, col_h,col_w;
                int col_step,col_height;

            };
        ///////////////////////////////////////
        //gemm
        //////////////////////////////////////
        class gemm{
            public:
            gemm(ssg_kernel &x):parent(x){
            }

            ~gemm(){
            }
            
            void gemm_init();
            void gemm_gpu();
            void gemm_cpu();

            cl_int err;
            ssg_kernel & parent;

            public:
            int img_n;
            int weight_h, weight_w;
            int col_height, col_step;
            int col_h,col_w;
            float * col_in;
            float * weight_in;
            float * b_in;
            float * gemm_out;

        };

        //////////////////////////////////////
        //maxpool
        /////////////////////////////////////
        class maxpool{
            public: 
                maxpool(ssg_kernel & x):parent(x){

                }
                ~maxpool(){

                }
                
                void maxpool_init();
                void maxpool_gpu();
                void maxpool_cpu();

            public:
                int col_n, col_c, col_h,col_w;
                int col_step, col_height;
                int f_n,f_c,f_h,f_w,f_size;
                int stride, pad;
                int img_n,img_c, img_h, img_w;

                float * pool_in;
                float * pool_out;
                cl_int err;
                ssg_kernel & parent;

        };
        ////////////////////////////////////////
        //kernel
        ///////////////////////////////////////

    public:
        ssg_kernel(float * img_in_,int img_w_,int img_h_,int img_c_, int img_n_,\
                float * w_in_,int weight_w_,int weight_h_,float *b_in_,float *img_out_)
            :Im2col(*this),Gemm(*this),Maxpool(*this),\
             img_w(img_w_),img_h(img_h_),img_c(img_c_),img_n(img_n_),\
             weight_w(weight_w_),weight_h(weight_h_)
         {
             img_in = new float[img_n*img_c*img_h*img_w];
             for(int iter=0;iter<img_n*img_c*img_h*img_w;iter++)
                 img_in[iter]= img_in_[iter];
   
             w_in = new float[weight_w*weight_h];
             for(int iter=0;iter<weight_h*weight_w;iter++)
                 w_in[iter] = w_in_[iter];

             b_in = new float[weight_w];
             for(int iter=0;iter<weight_w;iter++)
                    b_in[iter] = b_in_[iter];

             buf_img = NULL;
             buf_col =NULL;
             buf_weight = NULL;
             buf_bias = NULL;
             buf_gemm = NULL;
             buf_maxpool =NULL;
            
             col_in = NULL;

         }
        ~ssg_kernel()
        {
            //opencl mem
            if(buf_img !=NULL) clReleaseMemObject(buf_img);buf_img =NULL;
            if(buf_col !=NULL) clReleaseMemObject(buf_col);buf_col =NULL;
            if(buf_weight !=NULL) clReleaseMemObject(buf_weight);buf_weight=NULL;
            if(buf_bias !=NULL) clReleaseMemObject(buf_bias);buf_bias =NULL;
            if(buf_gemm !=NULL) clReleaseMemObject(buf_gemm);buf_gemm =NULL;
            if(buf_maxpool !=NULL) clReleaseMemObject(buf_maxpool);buf_maxpool =NULL;

            //c mem
            if(img_in != NULL) delete[] img_in; img_in=NULL;
            if(w_in != NULL) delete[] w_in; w_in=NULL;
            if(col_in !=NULL) delete [] col_in; col_in=NULL;
            if(b_in !=NULL) delete[] b_in ;b_in =NULL;
        }
        
        void get_cl_info();
        void init_device();
        void compile_program();
        void checkError(cl_int error, int line);

        public:
        //data buf
        float * img_in;
        float * col_in;
        float * w_in;
        float * b_in;
        
        //matrix size 
        int img_w, img_h,img_c,img_n;
        int col_w,col_h,col_c,col_n;
        int col_height,col_step;
        int f_n,f_c,f_h,f_w;
        int stride,pad;
        int weight_w, weight_h;

        //opencl device param
        cl_int err;
        cl_program program;
        cl_platform_id platform;
        cl_device_id device;
        cl_device_id devices[MAX_NUM_DEVICES];
        cl_uint numDevices;
        cl_context_properties props;
        cl_context context;
        cl_command_queue queue;
        cl_event event;

        //inner class
        im2col Im2col;
        gemm Gemm;
        maxpool Maxpool;
        
        //kernle buf
        cl_mem buf_img;
        cl_mem buf_col;
        cl_mem buf_weight;
        cl_mem buf_bias;
        cl_mem buf_gemm;
        cl_mem buf_maxpool;
};
       
