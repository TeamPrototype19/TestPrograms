#include <iostream>
#include <fstream>

#include "nntable_generated.h"

using namespace NNExecutor;

void flatbuff_serialize_test(std::string filename);
void flatbuff_deserialize_test(std::string filename);

int main(void) {
    std::cout << "This is flatbuffer test code.\n";

    flatbuff_serialize_test("fbuff_output.dat");
    flatbuff_deserialize_test("fbuff_output.dat");

    return 0;
}

void flatbuff_deserialize_test(std::string filename) {
    /* File open
     */
    std::ifstream fbfile;
    fbfile.open(filename.c_str(), std::ios::in | std::ios::binary);

    if( ! fbfile.is_open() )
        return;

    /* Get buffer size
     */
    fbfile.seekg(0, std::ios::end);
    int size = fbfile.tellg();
    fbfile.seekg(0, std::ios::beg);
    //std::cout << "File size = " << size << "\n";

    /* Read flatbuffer data file
     */
    char *buf = new char [ size ];
    fbfile.read((char*)buf, size);
    fbfile.close();

    
    /* Get a pointer to the root object insde the buffer
     */
    auto cgo = GetCompOutput( buf );

    // Get kernel handle
    auto insts = cgo->inst();
    auto inst_size = insts->Length();
    std::cout << "inst.size = " << inst_size << "\n";

    // Get opcode and opinfos
    for(int i = 0; i < inst_size; i++) {
        std::cout << "  + inst: " << i << "\n";
        auto kernel = insts->Get(i);
        auto opcode = kernel->opcode()->str();
        std::cout << "    name = " << opcode << "\n";
        auto opinfo = kernel->opinfo();
        if( kernel->opinfo_type() == kernel_info_Conv ) {
            //std::cout << "    type = " << "Conv" << "\n";
            auto opinfo_conv = static_cast<const Conv*>(opinfo);

            auto kernel_name = opinfo_conv->name()->str();
            auto kernel_size = opinfo_conv->kernel_size();
            auto stride_size = opinfo_conv->stride_size();
            auto pad_size    = opinfo_conv->pad_size();
            std::cout << "    kernel_size = " << kernel_size << "\n";
            std::cout << "    stride_size = " << stride_size << "\n";
            std::cout << "    pad_size    = " << pad_size << "\n";
        }
        else if( kernel->opinfo_type() == kernel_info_Relu ) {
            //std::cout << "    type = " << "Relu" << "\n";
            auto opinfo_relu = static_cast<const Relu*>(opinfo);

            auto kernel_name = opinfo_relu->name()->str();
            auto kernel_type = opinfo_relu->type()->str();
            std::cout << "    kernel_name = " << kernel_name << "\n";
            std::cout << "    kernel_type = " << kernel_type << "\n";
        }
    }

}

void flatbuff_serialize_test(std::string filename) {
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
    auto relu_type = builder.CreateString("Non-linear");

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
    fbfile.open(filename.c_str(), std::ios::out | std::ios::binary);
    fbfile.write((char*)buf_ptr, buf_size);
    fbfile.close();


}
