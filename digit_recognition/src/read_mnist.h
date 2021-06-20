#pragma once
#include "data.h"

int32_t get_int32_t(std::ifstream& file);

Data load_data(std::string images_path, std::string labels_path);
