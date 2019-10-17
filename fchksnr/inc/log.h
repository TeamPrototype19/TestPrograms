#ifndef _LOG_H_
#define _LOG_H_

#include <iostream>
#include <fstream>
#include <stdexcept>

#ifndef LOG_LEVEL
#define LOG_LEVEL 0
#endif

extern std::ofstream  logfs;
void open_log_file(std::string file_name);
void close_log_file(void);

#endif
