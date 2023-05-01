#ifndef _KR_CMDOPT_H_
#define _KR_CMDOPT_H_

#include "common/korangar.h"

typedef struct {
	char** values;
	i32    param_count;
} kr_param;

typedef struct {
	const char* option;
	i32			arg_count;
	i32		    length;
} kr_cmd_option;

#ifdef __cplusplus
extern "C" {
#endif

	kr_error
		kr_opt_get(i32 argc, char* argv[], const char* pattern, const char* arg, kr_param* param);

#ifdef __cplusplus
}
#endif

#endif /* _KR_CMDOPT_H_ */

