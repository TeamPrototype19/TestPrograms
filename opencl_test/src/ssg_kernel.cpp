#include "ssg_kernel.h"

void ssg_kernel::get_cl_info()
{
    opencl_info();
}

void ssg_kernel::init_device()
{
    
    program =NULL;
    platform=0;
    device =0;
    devices[MAX_NUM_DEVICES];
    numDevices =0;
    props = CL_CONTEXT_PLATFORM;
    context =0;
    queue =0;
    event =NULL;

    char deviceName[MAX_DEVICE_NAME];

    //connet GPU device 
    cl_platform_id * platformIds;
    platformIds = (cl_platform_id *) alloca(sizeof(cl_platform_id)*2);
    err= clGetPlatformIDs(2,platformIds,NULL);
    platform= platformIds[0];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    device = devices[CURRENT_DEVICE];
    props = (cl_context_properties)platform;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_DEVICE_NAME, deviceName, NULL);
    checkError(err, __LINE__);
    printf("## %d devices, running on %d: '%s'\n", numDevices, CURRENT_DEVICE, deviceName);

}

void ssg_kernel::compile_program()
{
    //Load source code
    FILE * fp_source, *fp_header;
    char * tmp_str;
    char * source_str, *header_str;
    size_t source_size, header_size;

    fp_header = fopen(CL_INCLUDE_FILE,"r");
    fp_source = fopen(CL_KERNEL_FILE,"r");

    if(!fp_source){
        fprintf(stderr,"Failed to load kernel\n");
        exit(1);
    }

    tmp_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size =fread(tmp_str,1,MAX_SOURCE_SIZE, fp_source);
    source_str = (char*)malloc(source_size+40);
    strncpy(source_str,tmp_str,source_size);

    header_size = fread(tmp_str,1,MAX_SOURCE_SIZE,fp_header);
    header_str = (char*)malloc(header_size+40);
    strncpy(header_str,tmp_str,header_size);

    size_t size_ = source_size+header_size+100;
    char * code_ = (char*)malloc(size_*sizeof(char));

    for(int c=0;c<size_;c++)
    {
        code_[c]='\0';
    }

    strcat(code_,header_str);
    strcat(code_,source_str);

    fclose(fp_header);
    fclose(fp_source);

    program = clCreateProgramWithSource(context,1,(const char **)&code_,(const size_t*)&size_,&err);

    free(code_);
    free(tmp_str);
    free(source_str);
    free(header_str);


    //build kernle
    err = clBuildProgram(program,1,&device,NULL,NULL,NULL);
    
    if(err !=0)
    {
        size_t len =0;
        clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&len);
        char * buildlog = (char*)calloc(len,sizeof(char));
        clGetProgramBuildInfo(program,device, CL_PROGRAM_BUILD_LOG,len,buildlog,NULL);
        printf("\n\nBuildlog: %s\n\n",buildlog);
        free(buildlog);
    }

    checkError(err,__LINE__);
}

void ssg_kernel::im2col::im2col_init(int f_n_,int f_h_,int stride_,int pad_,int mode)
{
    //gemm im2col
    //img : NCHW
    //filter: FNCFHFW
    //col: NFNOHOW

    img_n = parent.img_n;
    img_c = parent.img_c;
    img_h = parent.img_w;
    img_w = parent.img_h;
    
    f_h = f_w = f_h_;
    f_n = f_n_;
    stride = stride_;
    pad = pad_;

    col_n = img_n;

    //mode 0  gemm im2col
    //mode 1  pool im2col
    if(mode ==0)
        col_c = f_n;
    else
        col_c = img_c;

    col_h = (img_h+2*pad-f_h)/stride+1 ; // OH
    col_w = (img_w+2*pad-f_h)/stride+1 ; //OW
    
    //col = [img_n*OH*OW]x[f_h*f_w*img_c] // python
    //col = [f_h*f_w*img_c]x[img_n*OH*OW] // c
    //kernel :  [f_h*f_w*img_c]x[img_n*OH*OW]-->[img_n*OH*OW]x[f_h*f_w*img_c] 
    col_height = img_c *f_h*f_w;
    col_step = col_h * col_w * img_n;

    im2col_in = parent.img_in;
    im2col_out = new float[col_step*col_height];

}

float ssg_kernel::im2col::im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void ssg_kernel::im2col::im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}



void ssg_kernel::im2col::im2col_gpu()
{
    //gpu im2col
    char kernelname[100];
    sprintf(kernelname, "im2col_gpu_kernel");
    printf("%s\n",kernelname);

    cl_kernel kernel_im2col = clCreateKernel(parent.program,kernelname, &err);
    parent.checkError(err, __LINE__);

    parent.buf_img = clCreateBuffer(parent.context,CL_MEM_READ_ONLY,img_n*img_h*img_w*img_c*sizeof(float),NULL,&err);
    parent.buf_col = clCreateBuffer(parent.context,CL_MEM_READ_WRITE,col_height*col_step*sizeof(float),NULL,&err);
    err= clEnqueueWriteBuffer(parent.queue,parent.buf_img,CL_TRUE,0,img_n*img_h*img_w*img_c*sizeof(float),im2col_in,0,NULL,NULL);

    //set the arguments of the im2col
    err = clSetKernelArg(kernel_im2col,0,sizeof(int),(void*)&img_n);
    err = clSetKernelArg(kernel_im2col,1,sizeof(int),(void*)&img_c);
    err = clSetKernelArg(kernel_im2col,2,sizeof(int),(void*)&img_h);
    err = clSetKernelArg(kernel_im2col,3,sizeof(int),(void*)&img_w);
    err = clSetKernelArg(kernel_im2col,4,sizeof(int),(void*)&col_h);
    err = clSetKernelArg(kernel_im2col,5,sizeof(int),(void*)&col_w);
    err = clSetKernelArg(kernel_im2col,6,sizeof(int),(void*)&f_h);
    err = clSetKernelArg(kernel_im2col,7,sizeof(int),(void*)&pad);
    err = clSetKernelArg(kernel_im2col,8,sizeof(int),(void*)&stride);
    err = clSetKernelArg(kernel_im2col,9,sizeof(cl_mem),(void*)&parent.buf_img);
    err = clSetKernelArg(kernel_im2col,10,sizeof(cl_mem),(void*)&parent.buf_col);
    parent.checkError(err,__LINE__);

    //col_height = img_c * f_h*f_w
    //col_step = col_h*col_w * img_n
    const size_t local_im2col[2] = {f_h,col_h};
    const size_t global_im2col[2] = {(size_t)col_height,(size_t)col_h*col_w};

    err = clEnqueueNDRangeKernel(parent.queue,kernel_im2col,2,NULL,global_im2col,local_im2col,0,NULL,&parent.event);
    parent.checkError(err, __LINE__);
    err = clEnqueueReadBuffer(parent.queue,parent.buf_col,CL_TRUE,0,col_height*col_step*sizeof(float),im2col_out,0,NULL,NULL);
    parent.checkError(err, __LINE__);
   
#ifdef debug
    FILE * fp_col_gpu = fopen("./out/col_gpu_n2.txt","w");
    
    for(int iter=0;iter<col_height * col_step;iter++)
        fprintf(fp_col_gpu,"%1.4f\n",im2col_out[iter]);

    fclose(fp_col_gpu);
#endif

    //Free the memory objects
    clReleaseMemObject(parent.buf_img);parent.buf_img=NULL;
    
    if(parent.col_in !=NULL){
        delete [] parent.col_in;
        parent.col_in = NULL;
    }

    parent.col_in = im2col_out;
    parent.col_n = col_n;
    parent.col_c = col_c;
    parent.col_h = col_h;
    parent.col_w = col_w;

    //tranprot [N*OH*OW]x[C*FH*FW] -->[[C*FH*FW]x[N*OH*OW]
    parent.col_height = col_step;
    parent.col_step = col_height;
    parent.f_n = f_n;
    parent.f_c = f_c;
    parent.f_h = f_h;
    parent.f_w =f_w;
    parent.stride = stride;
    parent.pad = pad;

}

void ssg_kernel::im2col::im2col_cpu_()
{
    FILE * fp_tmp = fopen("./out/col_cpu_n2.txt","w");

    float * im2col_tmp = new float [col_height * col_h*col_w];
    int img_size = col_h * col_w;
    float * x_ptr = im2col_in;

    for(int img_num =0; img_num < img_n;img_num++)
    {
        im2col_cpu(x_ptr,img_c,img_h,img_w,f_h,stride,pad,im2col_tmp);

#ifdef debug
        for(int h=0;h<img_size;h++)
            for(int w=0;w<col_height;w++){
                //[OH*OW]x[CFH*FW]-->[CFH*FW]x[OH*OW]
                fprintf(fp_tmp,"%1.4f\n",im2col_tmp[w*img_size+h]);
            }
#endif
        x_ptr += img_h * img_w * img_c;
    }
    
    delete [] im2col_tmp;
    fclose(fp_tmp);
}


void ssg_kernel::gemm::gemm_init()
{
    weight_h   = parent.weight_h;
    weight_w   = parent.weight_w;
    col_height = parent.col_height;
    col_step   = parent.col_step;
    col_h = parent.col_h;
    col_w = parent.col_w;


    img_n      = parent.img_n;
    col_in     = parent.col_in;
    weight_in  = parent.w_in;
    b_in       = parent.b_in;
    gemm_out = new float[col_height*weight_w];

}

void ssg_kernel::gemm::gemm_gpu()
{
    //col[col_height*col_step]xW[weight_h * weight_w]=gemm_out[weight_h*col_step]
    char kernelname[100];
	sprintf(kernelname, "myGEMM%d", KERNEL);
	printf("%s\n",kernelname);

    cl_kernel kernel1 = clCreateKernel(parent.program, kernelname, &err);
	parent.checkError(err, __LINE__);

	// Prepare OpenCL memory objects
	parent.buf_weight = clCreateBuffer(parent.context, CL_MEM_READ_ONLY,  weight_h*weight_w*sizeof(float), NULL, &err);
	parent.buf_bias   = clCreateBuffer(parent.context, CL_MEM_READ_WRITE, weight_w * sizeof(float), NULL, &err);
	parent.buf_gemm   = clCreateBuffer(parent.context, CL_MEM_READ_WRITE, col_height*weight_w*sizeof(float), NULL, &err);

	float * subA = new float [col_height*col_step];
	float * subB = new float [weight_h * weight_w];
	cl_mem bufSubA = clCreateBuffer(parent.context, CL_MEM_READ_WRITE, col_height*col_step* sizeof(float), NULL, &err);
	cl_mem bufSubB = clCreateBuffer(parent.context, CL_MEM_READ_WRITE, weight_h * weight_w * sizeof(float), NULL, &err);
	parent.checkError(err, __LINE__);

	// Copy matrices to the GPU (also C to erase the results of the previous run)
	err = clEnqueueWriteBuffer(parent.queue, parent.buf_weight, CL_TRUE, 0, weight_h*weight_w*sizeof(float), weight_in, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(parent.queue, parent.buf_bias,   CL_TRUE, 0, weight_w*sizeof(float), b_in, 0, NULL, NULL);
	//set the arguments of the im2col
    err = clSetKernelArg(kernel1,0,sizeof(int),(void*)&col_height);
    err = clSetKernelArg(kernel1,1,sizeof(int),(void*)&col_step);
    err = clSetKernelArg(kernel1,2,sizeof(int),(void*)&weight_w);
    err = clSetKernelArg(kernel1,3,sizeof(int),(void*)&img_n);
    err = clSetKernelArg(kernel1,4,sizeof(cl_mem),(void*)&parent.buf_col);
    err = clSetKernelArg(kernel1,5,sizeof(cl_mem),(void*)&parent.buf_weight);
    err = clSetKernelArg(kernel1,6,sizeof(cl_mem),(void*)&parent.buf_bias);
    err = clSetKernelArg(kernel1,7,sizeof(cl_mem),(void*)&parent.buf_gemm);
    err = clSetKernelArg(kernel1,8,sizeof(cl_mem),(void*)&bufSubA);
    err = clSetKernelArg(kernel1,9,sizeof(cl_mem),(void*)&bufSubB);
    parent.checkError(err,__LINE__);

	// Configure the thread/work-group dimensions of the myGEMM kernel
#if KERNEL == 1 || KERNEL == 2
	const size_t local[2] = { col_w, 1 };
	const size_t global[2] = { (size_t)col_w*col_h, (size_t)weight_w };
#elif KERNEL ==3
	const size_t local[2] = { TS, TS / WPT };
	const size_t global[2] = { (size_t)M, (size_t)(N / WPT) };
#elif KERNEL == 6 
    const size_t local[2] = { TSM / WPTM, TSN / WPTN };
	const size_t global[2] = { (size_t)(M / WPTM), (size_t)(N / WPTN) };
#endif

    // Run the myGEMM kernel
	err = clEnqueueNDRangeKernel(parent.queue, kernel1, 2, NULL, global, local, 0, NULL, &parent.event);
    parent.checkError(err, __LINE__);


	// Copy the output matrix C back to the CPU memory
	err = clEnqueueReadBuffer(parent.queue, parent.buf_gemm, CL_TRUE, 0,  col_height*weight_w*sizeof(float),gemm_out, 0, NULL, NULL);
#ifdef debug
    err = clEnqueueReadBuffer(parent.queue, bufSubA,         CL_TRUE, 0,  col_height*col_step*sizeof(float), subA, 0, NULL, NULL);
	err = clEnqueueReadBuffer(parent.queue, bufSubB,         CL_TRUE, 0,  weight_h*weight_w*sizeof(float), subB, 0, NULL, NULL);
	parent.checkError(err, __LINE__);
    
    FILE * fp_subA = fopen("./out/subA.txt","w");
    FILE * fp_subB = fopen("./out/subB.txt","w");
    FILE * fp_gemm_gpu = fopen("./out/gemm_gpu_n2.txt","w");

    for(int h=0;h<col_height;h++){
        for(int w=0;w<col_step;w++){
            fprintf(fp_subA,"%1.4f\n",subA[h*col_step+w]);
        }               
    }
 
    for(int h=0;h<weight_h;h++){
        for(int w=0;w<weight_w;w++){
            fprintf(fp_subB,"%1.4f\n",subB[h*weight_w+w]);
        }               
    }

     for(int h=0;h<57600;h++){
        for(int w=0;w<30;w++){
            fprintf(fp_gemm_gpu,"%1.4f\n",gemm_out[h*30+w]);
        }               
    }

    fclose(fp_subA);
    fclose(fp_subB);
    fclose(fp_gemm_gpu);
#endif

    clReleaseMemObject(bufSubA);
    clReleaseMemObject(bufSubB);

    delete [] subA;
    delete [] subB;

    clReleaseMemObject(parent.buf_col);parent.buf_col=NULL;
    clReleaseMemObject(parent.buf_weight);parent.buf_weight=NULL;
    clReleaseMemObject(parent.buf_bias);parent.buf_bias=NULL;
    clReleaseMemObject(parent.buf_gemm);parent.buf_gemm=NULL;
   
    if(parent.img_in != NULL){
        delete [] parent.img_in;
        parent.img_in = NULL;
    }

    parent.img_in = gemm_out;
    parent.img_n = img_n;
    parent.img_c = weight_w;
    parent.img_h = parent.col_h;
    parent.img_w = parent.col_w;

}


void ssg_kernel::gemm::gemm_cpu()
{
    int col_ohow = col_height/img_n;
    int in_size = col_step * col_ohow;
    int out_size = col_ohow*weight_w;

    float * gemm_cpu_out = new float[col_height * weight_w];

    for(int iter_n=0;iter_n<img_n;iter_n++){
        for(int h=0;h<col_ohow;h++)
        {
            for(int w=0;w<weight_w;w++){
                float tmp_val =0;
                for(int k=0;k<col_step;k++)
                {
                    tmp_val+=col_in[h*col_step+k+iter_n*in_size]*weight_in[k*weight_w+w];
                }
                
                //+bias
                float out_val = tmp_val+b_in[w];

                //relu
                if(out_val <0)
                    out_val = 0.0f;

                gemm_cpu_out[w*col_ohow+h+iter_n*out_size] = out_val;
            }
        }
    }
#ifdef debug
    FILE * fp_tmp = fopen("./out/gemm_cpu_n2.txt","w");
    for(int iter=0;iter<col_height*weight_w;iter++)
        fprintf(fp_tmp,"%1.4f\n",gemm_cpu_out[iter]);

    fclose(fp_tmp);
#endif

    delete [] gemm_cpu_out;

}   
                
void ssg_kernel::maxpool::maxpool_init()
{
    col_height = parent.col_height;
    col_step = parent.col_step;
    col_h = parent.col_h;
    col_w = parent.col_w;
    f_h = f_w = parent.f_h;
    f_size = f_h*f_w;
    stride = parent.stride;
    pad = parent.pad;

    //img:NCHW
    //out:NCOHOW
    img_n  = parent.img_n;
    img_c  = parent.img_c;
    img_h  = parent.col_h;
    img_w  = parent.col_w;

    pool_in = parent.col_in;
    pool_out = new float[img_n*img_c*img_w*img_h];

}

void ssg_kernel::maxpool::maxpool_cpu()
{
    float * pool_out_cpu = new float[img_n*img_c*img_w*img_h];

    int in_step = f_size*img_c;
    int in_size = f_size*img_c*col_h*col_w;
    int out_step = img_c;
    int out_size = img_c*col_h*col_w; 
    int out_t_step = col_h*col_w;

    for(int iter_n=0;iter_n<img_n;iter_n++)
    {
        for(int iter_step=0; iter_step < in_size; iter_step+=f_size){
            float tmp_val = pool_in[iter_step+iter_n*in_size];
            for(int iter=1;iter<f_size;iter++){
                if(tmp_val<pool_in[iter_step+iter_n*in_size+iter])
                    tmp_val=pool_in[iter_step+iter_n*in_size+iter]; 
            }
            
            int row = (iter_step/f_size)/img_c;
            int col = (iter_step/f_size)%img_c;

            pool_out_cpu[col*out_t_step+row+iter_n*out_size] = tmp_val;

        }

    }

    delete [] pool_out_cpu;
#ifdef debug

    FILE * fp_tmp = fopen("./out/maxpool_cpu_n2.txt","w");
    for(int iter=0;iter<img_n*out_size;iter++)
        fprintf(fp_tmp,"%1.4f\n",pool_out_cpu[iter]);

    fclose(fp_tmp);
#endif

}

void ssg_kernel::maxpool::maxpool_gpu()
{
    char kernelname[100];
    sprintf(kernelname, "maxpool_gpu_kernel");
    printf("%s\n",kernelname);
    cl_kernel kernel_maxpool = clCreateKernel(parent.program,kernelname, &err);
    parent.checkError(err, __LINE__);

    parent.buf_maxpool = clCreateBuffer(parent.context,CL_MEM_READ_WRITE,img_n*img_c*img_h*img_w*sizeof(float),NULL,&err);

    //set the arguments of the im2col
    err = clSetKernelArg(kernel_maxpool,0,sizeof(int),(void*)&col_height);
    err = clSetKernelArg(kernel_maxpool,1,sizeof(int),(void*)&col_h);
    err = clSetKernelArg(kernel_maxpool,2,sizeof(int),(void*)&col_w);
    err = clSetKernelArg(kernel_maxpool,3,sizeof(int),(void*)&img_c);
    err = clSetKernelArg(kernel_maxpool,4,sizeof(int),(void*)&f_size);
    err = clSetKernelArg(kernel_maxpool,5,sizeof(cl_mem),(void*)&parent.buf_col);
    err = clSetKernelArg(kernel_maxpool,6,sizeof(cl_mem),(void*)&parent.buf_maxpool);
    parent.checkError(err,__LINE__);

    const size_t local_pool[2] = {img_h,1};
    const size_t global_pool[2] = {(size_t)img_h*img_w,(size_t)img_c};

    err = clEnqueueNDRangeKernel(parent.queue,kernel_maxpool,2,NULL,global_pool,local_pool,0,NULL,&parent.event);
    err = clEnqueueReadBuffer(parent.queue,parent.buf_maxpool,CL_TRUE,0,img_n*img_c*img_h*img_w*sizeof(float),pool_out,0,NULL,NULL);
    parent.checkError(err,__LINE__);

#ifdef debug
    FILE * fp_maxpool_gpu = fopen("./out/maxpool_gpu_n2.txt","w");

    for(int h=0;h<col_height;h++){
        for(int w=0;w<img_c;w++){
            fprintf(fp_maxpool_gpu,"%1.4f\n",pool_out[h*img_c+w]);
        }
    }

    fclose(fp_maxpool_gpu);
#endif

    //Free the memory objects
    clReleaseMemObject(parent.buf_col);parent.buf_col=NULL;
    clReleaseMemObject(parent.buf_maxpool);parent.buf_maxpool=NULL;

    if(parent.img_in !=NULL){
        delete [] parent.img_in;
        parent.img_in = NULL;
    }

    parent.img_in = pool_out;

}

                
                
                
                
void ssg_kernel::checkError(cl_int error, int line) {
	if (error != CL_SUCCESS) {
		switch (error) {
		case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
		case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
		case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
        case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
		case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
		case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
		case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
		case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
		case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
		case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
		case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
		case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
		case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
		case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
		case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
		case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
		case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
		case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
		case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
		case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
		case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
		case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
		case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
		case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
		case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
		case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
		case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
		case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
		case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
		case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
		case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
		case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
		case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
		case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
		case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
		case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
		case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
		case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
		case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
		case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
		case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
		case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
		case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
		case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
		case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
		case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
		case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
		case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
		case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
		case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
		case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
		case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
		case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
		case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
		case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
		case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
		case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
		case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
		case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
		case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
		default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
		}
		exit(1);
	}
}






