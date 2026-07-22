#pragma once

#ifdef _WIN32
  #ifdef SCANNER_BUILD_DLL
    #define SCANNER_API __declspec(dllexport)
  #elif defined(SCANNER_USE_DLL)
    #define SCANNER_API __declspec(dllimport)
  #else
    #define SCANNER_API
  #endif
#else
  #define SCANNER_API __attribute__((visibility("default")))
#endif
