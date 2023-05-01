#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>

#define KR_LOG_ENABLE 1
#define KR_LOG_TRACE_LEVEL 0
#define KR_LOG_INFO_LEVEL 1

kr_error 
kr_log(const char* format, ... ) {
  time_t timer;
  char buffer[26] = { 0 };
    struct tm* tm_info;

    timer = time(NULL);
    tm_info = localtime(&timer);

    strftime(buffer, 25, "%Y-%m-%d %H:%M:%S", tm_info);
  #if KR_LOG_ENABLE
    va_list arglist;
#ifdef KR_LOG_TRACE_LEVEL
    printf( "[TRACE %s] ", buffer );
#endif
    va_start( arglist, format );
    vprintf( format, arglist );
    va_end( arglist );
#endif

  return kr_success;
}
