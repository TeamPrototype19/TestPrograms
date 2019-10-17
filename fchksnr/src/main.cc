#include <unistd.h>
//#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "log.h"
//#include "nnexec.hpp"

void readBinaryData(char* &buf, int &size, std::string filename);
void float_snr_check( std::ofstream &rfs, float *in, float *out, int data_size);
//void writeBinaryData(char* &buf, int &size, std::string filename);

int main(int argc, char **argv) {
	char option;
	const char *optstring = "r:i:o:";

    open_log_file("log.txt");

    /* Parsing Command arguments
     */
	if( argc != 7 ) {
        std::cerr << "Not enough input arguments" << std::endl;
        std::cerr << "nne -i [input file name] -o [output file name] -r [report file name]" << std::endl;
        logfs << "--> invalid command arguments.\n";
		return -1;
	}

    std::string inputFileName  = "input.dat";
    std::string outputFileName = "output.dat";
    std::string reportFileName = "report.txt";

    while( -1 != (option = getopt(argc, argv, optstring))) {
		switch(option) {
			case 'r':	reportFileName = optarg;
						break;
			case 'i':	inputFileName = optarg;
						break;
			case 'o':	outputFileName = optarg;
						break;
		}
	}

    std::cout << "Input  file path = " << inputFileName << "\n";
    std::cout << "Output file path = " << outputFileName << "\n";
    std::cout << "Report file path = " << reportFileName << "\n";

    /* Report file open 
     */
    std::ofstream rfs;
    rfs.open(reportFileName.c_str(), std::ios::out);

    if( ! rfs.is_open() ) {
        std::cerr << "[ERROR] can't open file: " << reportFileName << "\n";
        return -1;
    }


    /* Main processing
     */
    try {
        char *ibuf = nullptr, *obuf = nullptr;
        int ibsize, obsize;


        // Read input/output file size
        readBinaryData( ibuf, ibsize, inputFileName );
        readBinaryData( obuf, obsize, outputFileName );

        if( ibsize != obsize )
            throw std::runtime_error("The size of input and output files is not same.");


        std::cout << "Checking...\n";
        float_snr_check( rfs, (float*)ibuf, (float*)obuf, ibsize / sizeof(float) );
    }
    catch (const std::exception& e) {
        std::cout << e.what() << "\n";
        std::cout << "Program receives exception. Program will be terminated.\n";
    }

    std::cout << "Finished!\n";

    rfs.close();

    close_log_file();

	return 0;
}

void readBinaryData(char* &buf, int &size, std::string filename) {
    /* File open
     */
    size = 0;
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ios::in | std::ios::binary);

    if( ! ifs.is_open() ) {
        std::cerr << "[ERROR] can't open file " << filename << "\n";
        throw std::runtime_error("Program will be terminated.\n");
        return;
    }

    /* Get buffer size
     */
    ifs.seekg(0, std::ios::end);
    size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if( size == 0 )
        return;

    /* Buffer allocation
     */
    buf = new char [ size ];

    ifs.read((char*)buf, size);
    ifs.close();

    return;
}

void float_snr_check( 
    std::ofstream &rfs, 
    float *in, 
    float *out, 
    int data_size
)
{
    float sigPw = 0;
    float nosPw = 0;

    for(int i = 0 ; i < data_size ; i++) {
        float diff = *in - *out;
        if( *in != *out )
            rfs << "[" << i << "]\t" << diff << "\t" << *in << "\t" << *out << "\n";

        sigPw += (*in) * (*in);
        nosPw += diff * diff;
    }

    rfs << "Final SNR = " << 10*log10(sigPw / nosPw);

    return;
}


#if 0
void writeBinaryData(char* &buf, int &size, std::string filename) {
    /* File open
     */
    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ios::out | std::ios::binary);

    if( ! ofs.is_open() ) {
        std::cerr << "[ERROR] can't open file " << filename << "\n";
        throw runtime_error("Program will be terminated.\n");
        return;
    }

    ofs.write( buf, size );
    ofs.close();

    return;
}
#endif
