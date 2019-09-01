#include <iostream>
#include <fstream>

#include "nntable_generated.h"

using namespace NNExecutor;

int main(void) {
    std::cout << "This is flatbuffer test code.\n";

    flatbuffers::FlatBufferBuilder builder(128);

    /* Convolution OP code generation
     */
    auto conv_name = builder.CreateString("ConvK1S1P1");
    int  conv_kernel_size = 1;
    int  conv_stride_size = 1;
    int  conv_pad_size = 0;

    auto op_conv = CreateConv(builder, conv_name, conv_kernel_size,
            conv_stride_size, conv_pad_size);

    /* Relu OP code generation
     */
    auto relu_name = builder.CreateString("Relu");
    auto relu_type = builder.CreateString("None");

    auto op_relu = CreateRelu(builder, relu_name, relu_type);

    /* Table Kernel generation
     */
    auto conv_opcode = builder.CreateString("CpuConv");
    auto kernel_conv = CreateKernel(builder, conv_opcode, kernel_info_Conv, op_conv.Union());
    auto relu_opcode = builder.CreateString("CpuRelu");
    auto kernel_relu = CreateKernel(builder, relu_opcode, kernel_info_Relu, op_relu.Union());
    
    std::vector<flatbuffers::Offset<Kernel>> kernel_vector;
    kernel_vector.push_back( kernel_conv );
    kernel_vector.push_back( kernel_relu );
    auto kernels = builder.CreateVector( kernel_vector );

    /* Finally, create CompOutput table
     */
    auto cgo = CreateCompOutput(builder, kernels );

    builder.Finish( cgo );


    /* Write buffer contents (serialization)
     */
    uint8_t *buf_ptr = builder.GetBufferPointer();
    int buf_size = builder.GetSize();

    std::ofstream fbfile;
    fbfile.open("fbuff_output.dat", std::ios::out | std::ios::binary);
    fbfile.write((char*)buf_ptr, buf_size);
    fbfile.close();
}
