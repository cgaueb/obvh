#include "integrator.h"
#include "util.h"

#include <stdio.h>

#define KR_FAILED_TO_LOAD_INTEGRATOR ((kr_error)"Failed to load integrator DLL")
#define KR_FAILED_TO_LOAD_INTEGRATOR_SYMBOL ((kr_error)"Failed to load integrator symbol from DLL")
kr_error kr_integrator_load(kr_integrator* integrator, const char* name) {
	char library_name[128] = { 0 };
#if KR_LINUX
	sprintf(library_name, "./lib%s.so", name);
#elif KR_WIN32 /* KR_LINUX */
#ifdef KR_DEBUG
	sprintf(library_name, "%s.dll", name);
#else
	sprintf(library_name, "%s.dll", name);
#endif /* KR_DEBUG */
#endif /* KR_WIN32 */
	const kr_handle library = kr_dlopen(library_name);
	if(!library)
		return KR_FAILED_TO_LOAD_INTEGRATOR;

	const kr_handle symbol = kr_dlsym(library, "korangar_action");
	if (!symbol) {
		kr_dlclose(library);
		return KR_FAILED_TO_LOAD_INTEGRATOR_SYMBOL;
	}

	integrator->callback = symbol;
	integrator->context  = (kr_handle) integrator->callback(kr_null, kr_null, KR_ACTION_CREATE);
	integrator->library  = library;
	
	return kr_success;
}

kr_error kr_integrator_call(kr_integrator* integrator, kr_handle descriptor, kr_action action) {
	return integrator->callback(integrator, descriptor, action);
}