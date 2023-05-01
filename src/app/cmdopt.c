#include "common/korangar.h"
#include "cmdopt.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

kr_internal b32 
kr_arg_match(const char* arg, const char* pattern, kr_cmd_option* opt) {
	if (opt) 
		opt->arg_count = -1;

	if (*arg == '-') arg++;

	const char* pattern_beg = pattern;
	const char* pattern_end = pattern;
	intptr_t    pattern_len = pattern_end - pattern_beg;
	i32 current_arg_count = 0;
	while (*pattern_beg) {
		while (*pattern_end && *pattern_end != '[') {
			pattern_end++;
		}

		pattern_len = pattern_end - pattern_beg;
		current_arg_count = (*pattern_end == '[') ? *(++pattern_end) - '0' : -1;
		if (strncmp(arg, pattern_beg, pattern_len) != 0) {
			if (current_arg_count == -1) return kr_false;
			while (*pattern_beg && *pattern_beg != ']') {
				pattern_beg++;
			}
			pattern_beg++;
			continue;
		}
		if (opt) {
			opt->arg_count = current_arg_count;
			opt->option = pattern_beg;
			opt->length = (i32)pattern_len;
		}
		return kr_true;
	}

	return kr_false;
}

kr_error
kr_opt_get(i32 argc, char* argv[], const char* pattern, const char* arg, kr_param* param) {
	i32 at = -1;
	i32 arg_count = 0;
	param->values = kr_null;
	param->param_count = 0;
	kr_cmd_option opt = { 0 };
	kr_cmd_option match_opt = { 0 };
	b32 match = kr_arg_match(arg, pattern, &opt);
	if (!match) return kr_false;

	for (i32 i = 1; i < argc; i++) {
		b32 match = kr_arg_match(argv[i], arg, kr_null);
		if (!match) continue;
		at = i;
		match_opt = opt;
		break;
	}
	if (at < 0) return kr_null;

	param->values = &argv[at];
	if (match_opt.arg_count > 0) {
		param->param_count = match_opt.arg_count;
	}
	return kr_success;
}