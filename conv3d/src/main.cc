#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <cmath>

#include "type.h"
#include "conv3d.h"

void check_sqnr( float *ref, float *out, int out_num, float &sqnr ) {
    float spower = 0;   // signal power
    float npower = 0;   // noise power

    for(int i = 0 ; i < out_num ; i++) {
        float signal = (*ref);
        float noise = (*ref++) - (*out++);
        spower += signal * signal;
        npower += noise * noise;
    }

    sqnr = 10 * log10( spower / npower );
    //std::cout << "spower = " << spower << "\t";
    //std::cout << "npower = " << npower << "\n";

    return;
}

int readBinaryData(char* &buf, std::string filename) {
    /* File open
     */
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ios::in | std::ios::binary);

    if( ! ifs.is_open() )
        return 0;

    /* Get buffer size
     */
    ifs.seekg(0, std::ios::end);
    int size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if( size == 0 )
        return 0;

    /* Buffer allocation
     */
    buf = new char [ size ];

    ifs.read((char*)buf, size);
    ifs.close();

    return size;
}

int main(int argc, char **argv) {
	char option;
	const char *optstring = "i:o:w:b:o:";

	if( argc != 9 ) {
        std::cerr << "Not enough inputs." << std::endl;
        std::cerr << "nnc -i [IFM file] -w [Weight file] -b [bias file] -o [OFM file]" << std::endl;
		return -1;
	}

    std::string ifmFileName    = "ifm.dat";
	std::string weightFileName = "weight.dat";
	std::string biasFileName   = "bias.dat";
	std::string ofmFileName    = "ofm.dat";

    while( -1 != (option = getopt(argc, argv, optstring))) {
		switch(option) {
			case 'i':	ifmFileName = optarg;
						break;
			case 'w':	weightFileName = optarg;
						break;
			case 'b':	biasFileName = optarg;
						break;
			case 'o':	ofmFileName = optarg;
						break;
		}
	}

    /* Display the target files
     */
    std::cout << "IFM file name    = " << ifmFileName << std::endl;
    std::cout << "Weight file name = " << weightFileName << std::endl;
    std::cout << "Bias file name   = " << biasFileName << std::endl;
    std::cout << "OFM file name    = " << ofmFileName << std::endl;

    char *ibuf, *wbuf, *bbuf, *obuf, *robuf;

    int ibuf_size = readBinaryData( ibuf, ifmFileName );
    int wbuf_size = readBinaryData( wbuf, weightFileName );
    int bbuf_size = readBinaryData( bbuf, biasFileName );
    int obuf_size = readBinaryData( robuf, ofmFileName );
    if( ibuf_size < 1 ) {
        std::cerr << "[Error] read input file size is zero!" << std::endl;
        return 0;
    }
    if( wbuf_size < 1 ) {
        std::cerr << "[Error] read weight file size is zero!" << std::endl;
        return 0;
    }
    if( bbuf_size < 1 ) {
        std::cerr << "[Error] read bias file size is zero!" << std::endl;
        return 0;
    }
    if( obuf_size < 1 ) {
        std::cerr << "[Error] read output file size is zero!" << std::endl;
        return 0;
    }

    obuf = new char [ obuf_size ];
    memset( obuf, 0, obuf_size );


    /* construct conv test date information 
     */
    ConvInfo cinfo;
    cinfo.ifmDim[0] = 1;
    cinfo.ifmDim[1] = 1; 
    cinfo.ifmDim[2] = 28;
    cinfo.ifmDim[3] = 28;
    cinfo.kernel_size_w = 5; cinfo.kernel_size_h = 5;
    cinfo.stride_size_w = 1; cinfo.stride_size_h = 1;
    cinfo.pad_size_w    = 0; cinfo.pad_size_h    = 0;
    cinfo.output_num = 20;

    /* Run conv operation test code 
     */
    test_kernel_conv3d( 
        (float*) obuf,
        (float*) ibuf,
        (float*) wbuf,
        (float*) bbuf,
        cinfo
    );


    /* Check the SQNR of the output
     */
    float Sqnr;
    check_sqnr( (float*)robuf, (float*)obuf, obuf_size/sizeof(float), Sqnr );

    std::cout << "SQNR = " << Sqnr << std::endl;

    /* Free allocated buffers
     */
    delete [] ibuf;
    delete [] wbuf;
    delete [] bbuf;
    delete [] obuf;
    delete [] robuf;

    return 0;
}

