#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstring>
#include <cmath>

#include "type.h"
#include "batchnorm.h"

void check_sqnr( float *ref, float *out, int out_num, float &sqnr ) {
    float spower = 0;   // signal power
    float npower = 0;   // noise power

    for(int i = 0 ; i < out_num ; i++) {
        float signal = (*ref);
        float noise = (*ref++) - (*out++);
        spower += signal * signal;
        npower += noise * noise;
        
#if 1
        if( fabs(noise) > 0.000001 ) {
            std::cout << "out_num = " << out_num << std::endl;
            std::cout << "[" << i << "]  "    \
                      << "ref = " << *(ref-1) \
                      << "\tout = " << *(out-1) \
                      << "\tdiff = " << noise << std::endl;
            return;
        }
#endif
    }

    sqnr = 10 * log10( spower / npower );
    //std::cout << "spower = " << spower << "\t";
    //std::cout << "npower = " << npower << "\n";

    return;
}

int readConfigFile( BatchNormInfo& rinfo, std::string filename ) {
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

        if( key.compare("ifm.size.n") == 0 ) {
            iss >> value;
            rinfo.ifmDim[0] = value; 
        }
        else if( key.compare("ifm.size.c") == 0 ) {
            iss >> value;
            rinfo.ifmDim[1] = value; 
        }
        else if( key.compare("ifm.size.h") == 0 ) {
            iss >> value;
            rinfo.ifmDim[2] = value; 
        }
        else if( key.compare("ifm.size.w") == 0 ) {
            iss >> value;
            rinfo.ifmDim[3] = value; 
        }
        else if( key.compare("eps") == 0 ) {
            float value_f;
            iss >> value_f;
            rinfo.eps = value_f;
            std::cout << "eps = " << rinfo.eps << "\n";
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
	const char *optstring = "i:o:c:m:v:s:";

	if( argc != 13 ) {
        std::cerr << "Not enough inputs." << std::endl;
        std::cerr << "nnc -i [IFM file] -o [OFM file] -c [config file] -m [mean] -v [var] -s [scale]" << std::endl;
		return -1;
	}

    std::string ifmFileName    = "ifm.dat";
	std::string ofmFileName    = "ofm.dat";
    std::string configFileName = "config.txt";
    std::string meanFileName   = "mean.dat";
    std::string varFileName    = "var.dat";
    std::string scaleFileName  = "scale.dat";

    while( -1 != (option = getopt(argc, argv, optstring))) {
		switch(option) {
			case 'i':	ifmFileName = optarg;
						break;
			case 'o':	ofmFileName = optarg;
						break;
			case 'c':	configFileName = optarg;
						break;
			case 'm':	meanFileName = optarg;
						break;
			case 'v':	varFileName = optarg;
						break;
			case 's':	scaleFileName = optarg;
						break;
		}
	}

    /* Display the target files
     */
    std::cout << "IFM file name    = " << ifmFileName << std::endl;
    std::cout << "OFM file name    = " << ofmFileName << std::endl;
    std::cout << "config file name = " << configFileName << std::endl;
    std::cout << "mean file name   = " << meanFileName << std::endl;
    std::cout << "var file name    = " << varFileName << std::endl;
    std::cout << "scale file name  = " << scaleFileName << std::endl;

    char *ibuf, *obuf, *robuf, *vbuf, *mbuf, *sbuf;

    int ibuf_size = readBinaryData( ibuf, ifmFileName );
    int obuf_size = readBinaryData( robuf, ofmFileName );
    int mbuf_size = readBinaryData( mbuf, meanFileName );
    int vbuf_size = readBinaryData( vbuf, varFileName );
    int sbuf_size = readBinaryData( sbuf, scaleFileName );
    if( ibuf_size < 1 ) {
        std::cerr << "[Error] read input file size is zero!" << std::endl;
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
    BatchNormInfo rinfo;
    if( readConfigFile( rinfo, configFileName ) != 0 ) {
        std::cerr << "[Error] fail to process config file!" << std::endl;
        return 0;
    }


    /* Run conv operation test code 
     */
    test_kernel_batchnorm( 
        (float*) obuf,
        (float*) ibuf,
        (float*) mbuf,
        (float*) vbuf,
        (float*) sbuf,
        rinfo
    );


    /* Check the SQNR of the output
     */
    float Sqnr;
    check_sqnr( (float*)robuf, (float*)obuf, obuf_size/sizeof(float), Sqnr );

    std::cout << "SQNR = " << Sqnr << " (dB)" << std::endl;

    /* Free allocated buffers
     */
    delete [] ibuf;
    delete [] obuf;
    delete [] robuf;

    return 0;
}

