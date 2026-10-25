#include "core/direct_storage_reader.h"

#include <exception>
#include <iostream>

extern "C" __declspec(dllexport) int ninfer_directstorage_backend_probe() {
    try {
        auto reader = ninfer::DirectStorageReader::open();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Backend DirectStorage reader: " << error.what() << '\n';
        return 1;
    }
}
