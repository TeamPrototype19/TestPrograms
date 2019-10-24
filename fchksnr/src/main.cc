#include <unistd.h>
//#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iomanip>

#include "log.h"
//#include "nnexec.hpp"

void readBinaryData(char* &buf, int &size, std::string filename);
void float_snr_check( std::ofstream &rfs, float *in, float *out, int data_size, std::string);
//void writeBinaryData(char* &buf, int &size, std::string filename);

int main(int argc, char **argv) {
	char option;
	const char *optstring = "t:r:R:n:";

    open_log_file("log.txt");

    /* Parsing Command arguments
     */
	if( argc != 7 && argc != 9 ) {
        std::cerr << "Not enough input arguments" << std::endl;
        std::cerr << "nne -t [test file name] -r [reference file name] -R [report file name]\n";
        std::cerr << "(optinal): -n [layer name]\n";
        logfs << "--> invalid command arguments.\n";
		return -1;
	}

    std::string inputFileName  = "input.dat";
    std::string referFileName  = "refer.dat";
    std::string reportFileName = "report.txt";
    std::string layerName;

    while( -1 != (option = getopt(argc, argv, optstring))) {
		switch(option) {
			case 'R':	reportFileName = optarg;
						break;
			case 't':	inputFileName = optarg;
						break;
			case 'r':	referFileName = optarg;
						break;
			case 'n':	layerName = optarg;
						break;
		}
	}

#ifdef VERBOSE
    std::cout << "Input  file path = " << inputFileName << "\n";
    std::cout << "Refer  file path = " << referFileName << "\n";
    std::cout << "Report file path = " << reportFileName << "\n";
    if( layerName.length() > 0 )
        std::cout << "Layer name       = " << layerName << "\n";
#endif

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
        char *ibuf = nullptr, *rbuf = nullptr;
        int ibsize, rbsize;


        // Read input/output file size
        readBinaryData( ibuf, ibsize, inputFileName );
        readBinaryData( rbuf, rbsize, referFileName );

        if( ibsize != rbsize )
            throw std::runtime_error("[ERROR] The size of input and output files is not same.");


#ifdef VERBOSE
        std::cout << "Checking...\n";
#endif
        float_snr_check( rfs, (float*)rbuf, (float*)ibuf, ibsize / sizeof(float), layerName );
    }
    catch (const std::exception& e) {
        std::cout << e.what() << "\n";
        std::cout << "Program receives exception. Program will be terminated.\n";
    }

#ifdef VERBOSE
    std::cout << "Finished!\n";
#endif

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
    float *ref, 
    float *test, 
    int data_size,
    std::string layerName
)
{
    float sigPw = 0;
    float nosPw = 0;
    int   diff_cnt = 0;

    for(int i = 0 ; i < data_size ; i++) {
        float diff = *ref - *test;
        if( *ref != *test) {
            int r = *(int*)ref;
            int t = *(int*)test;
            rfs << std::scientific;
            rfs << "[" << i << "]\t" << diff << "\tRef: " << *ref << "\tTst: " << *test << "\t";
            rfs << "r.hex = 0x" << std::setfill('0') << std::right << std::setw(8) << std::hex << r << "\t";
            rfs << "t.hex = 0x" << std::setfill('0') << std::right << std::setw(8) << t << "\n";
            rfs << std::setfill(' ') << std::dec;   // roll-back
            rfs << std::defaultfloat;
            diff_cnt++;
        }

        sigPw += (*ref) * (*ref);
        nosPw += diff * diff;
        
        ref++;
        test++;
    }

    rfs << "/*-----------------------------------------------------------------*\n";
    rfs << " * SNR CHECK REPORT                                                *\n";
    rfs << " * Checked data number = " << std::setw(7) << std::right << data_size << "                                   *\n";
    rfs << " * NotSame data number = " << std::setw(7) << std::right << diff_cnt  << "                                   *\n";
    rfs << " * Final SNR           = " << std::setw(7) << std::right << 10*log10(sigPw / nosPw) << " (dB)";
    if( layerName.length() > 0)
        rfs << "    [" << std::setw(20) << std::left << layerName << "]   ";
    else
        rfs << "                             ";
    rfs << " *\n";
    rfs << " *-----------------------------------------------------------------*/\n";

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
