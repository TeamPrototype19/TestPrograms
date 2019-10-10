#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstring>
#include <cmath>

#include "type.h"
#include "softmax.h"

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

int readConfigFile( SoftmaxInfo& sinfo, std::string filename ) {
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
            sinfo.ifmDim[0] = value; 
        }
        else if( key.compare("ifm.size.c") == 0 ) {
            sinfo.ifmDim[1] = value; 
        }
        else if( key.compare("ifm.size.h") == 0 ) {
            sinfo.ifmDim[2] = value; 
        }
        else if( key.compare("ifm.size.w") == 0 ) {
            sinfo.ifmDim[3] = value; 
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
	const char *optstring = "i:o:c:";

	if( argc != 7 ) {
        std::cerr << "Not enough inputs." << std::endl;
        std::cerr << "nnc -i [IFM file] -o [OFM file] -c [config file]" << std::endl;
		return -1;
	}

    std::string ifmFileName    = "ifm.dat";
	std::string ofmFileName    = "ofm.dat";
    std::string configFileName = "config.txt";

    while( -1 != (option = getopt(argc, argv, optstring))) {
		switch(option) {
			case 'i':	ifmFileName = optarg;
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
    std::cout << "OFM file name    = " << ofmFileName << std::endl;
    std::cout << "config file name = " << configFileName << std::endl;

    char *ibuf, *obuf, *robuf;

    int ibuf_size = readBinaryData( ibuf, ifmFileName );
    int obuf_size = readBinaryData( robuf, ofmFileName );
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
    SoftmaxInfo sinfo;
    if( readConfigFile( sinfo, configFileName ) != 0 ) {
        std::cerr << "[Error] fail to process config file!" << std::endl;
        return 0;
    }


    /* Run conv operation test code 
     */
    test_kernel_softmax( 
        (float*) obuf,
        (float*) ibuf,
        sinfo
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

