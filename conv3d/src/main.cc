#include <iostream>
#include <fstream>
#include <sstream>
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
        
#if 0
        if( fabs(noise) > 0.000001 ) {
            std::cout << "ref = " << *(ref-1) \
                      << "\tout = " << *(out-1) \
                      << "\tdiff = " << noise << std::endl;
        }
#endif
    }

    sqnr = 10 * log10( spower / npower );
    //std::cout << "spower = " << spower << "\t";
    //std::cout << "npower = " << npower << "\n";

    return;
}

int readConfigFile( ConvInfo& cinfo, std::string filename ) {
    /* File open
     */
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ios::in );

    if( ! ifs.is_open() ) {
        std::cout << "[ERROR] File open error: " << filename << std::endl;
        return -1;
    }

    std::string line, key;
    int value;
    while( std::getline( ifs, line ) ) {
        std::istringstream iss( line );

        iss >> key;
        iss >> value;

        if( key.compare("ifm.size.n") == 0 ) {
            cinfo.ifmDim[0] = value; 
        }
        else if( key.compare("ifm.size.c") == 0 ) {
            cinfo.ifmDim[1] = value; 
        }
        else if( key.compare("ifm.size.h") == 0 ) {
            cinfo.ifmDim[2] = value; 
        }
        else if( key.compare("ifm.size.w") == 0 ) {
            cinfo.ifmDim[3] = value; 
        }
        else if( key.compare("kernel.size.w") == 0 ) {
            cinfo.kernel_size_w = value;
        }
        else if( key.compare("kernel.size.h") == 0 ) {
            cinfo.kernel_size_h = value;
        }
        else if( key.compare("stride.size.w") == 0 ) {
            cinfo.stride_size_w = value;
        }
        else if( key.compare("stride.size.h") == 0 ) {
            cinfo.stride_size_h = value;
        }
        else if( key.compare("pad.size.w") == 0 ) {
            cinfo.pad_size_w = value;
        }
        else if( key.compare("pad.size.h") == 0 ) {
            cinfo.pad_size_h = value;
        }
        else if( key.compare("group_num") == 0 ) {
            cinfo.group_num = value;
        }
        else if( key.compare("output_num") == 0 ) {
            cinfo.output_num = value;
        }
        else {
            // invalid parameter !!
            std::cerr << "[ERROR] invalid config parameter: " << key << std::endl;
            return -1;
        }
    }

    return 0;
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
	const char *optstring = "i:o:w:b:c:";

	if( argc != 11 ) {
        std::cerr << "Not enough inputs." << std::endl;
        std::cerr << "nnc -i [IFM file] -w [Weight file] -b [bias file] -o [OFM file] -c [config file]" << std::endl;
		return -1;
	}

    std::string ifmFileName    = "ifm.dat";
	std::string weightFileName = "weight.dat";
	std::string biasFileName   = "bias.dat";
	std::string ofmFileName    = "ofm.dat";
    std::string configFileName = "config.txt";
    int bs_n, bs_c, bs_h, bs_w;
    int kernel_size, stride_size, pad_size;
    int output_num, group_num;

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
			case 'c':	configFileName = optarg;
						break;
		}
	}

    /* Display the target files
     */
    std::cout << "IFM file name    = " << ifmFileName << std::endl;
    std::cout << "Weight file name = " << weightFileName << std::endl;
    std::cout << "Bias file name   = " << biasFileName << std::endl;
    std::cout << "OFM file name    = " << ofmFileName << std::endl;
    std::cout << "config file name = " << configFileName << std::endl;

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
    if( readConfigFile( cinfo, configFileName ) != 0 ) {
        std::cerr << "[Error] fail to process config file!" << std::endl;
        return 0;
    }


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

    std::cout << "SQNR = " << Sqnr << " (dB)" << std::endl;

    /* Free allocated buffers
     */
    delete [] ibuf;
    delete [] wbuf;
    delete [] bbuf;
    delete [] obuf;
    delete [] robuf;

    return 0;
}

