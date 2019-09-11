#include <iostream>
#include <fstream>
#include <unistd.h>

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

    char *ibuf, *wbuf, *bbuf, *obuf;

    if( readBinaryData( ibuf, ifmFileName ) < 1 )
        std::cerr << "[Warning] read file size is zero!" << std::endl;
    if( readBinaryData( wbuf, weightFileName ) < 1 )
        std::cerr << "[Warning] read file size is zero!" << std::endl;
    if( readBinaryData( bbuf, biasFileName ) < 1 )
        std::cerr << "[Warning] read file size is zero!" << std::endl;
    if( readBinaryData( obuf, ofmFileName ) < 1 )
        std::cerr << "[Warning] read file size is zero!" << std::endl;


    /* Free allocated buffers
     */
    delete [] ibuf;
    delete [] wbuf;
    delete [] bbuf;
    delete [] obuf;

    return 0;
}

