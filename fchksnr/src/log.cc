#include "log.h"

std::ofstream  logfs;

void open_log_file(std::string file_name) {
    logfs.open(file_name);
    if( ! logfs.is_open() )
        throw std::runtime_error("Can't open log file!\nProgram will be terminated!");
}

void close_log_file(void) {
    logfs.close();
}
