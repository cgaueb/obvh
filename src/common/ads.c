#include "ads.h"
#include "util.h"

#include <stdio.h>

kr_error kr_ads_load(kr_ads* ads, const char* name) {
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

	if (!library)
		return KR_ERROR("failed to load library");

	const kr_handle symbol = kr_dlsym(library, "korangar_ads_action");
	if (!symbol) {
		kr_dlclose(library);
		return KR_ERROR("failed to load symbol (korangar_ads_action)");
	}

	ads->callback = symbol;
	ads->context  = (kr_handle) ads->callback(kr_null, kr_null, KR_ADS_ACTION_CREATE);
	ads->library  = library;

	return kr_success;
}

kr_error kr_ads_call(kr_ads* ads, kr_handle descriptor, kr_ads_action action) {
	return ads->callback(ads, descriptor, action);
}
