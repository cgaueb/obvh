#ifndef _KORANGAR_INTEGRATOR_H_
#define _KORANGAR_INTEGRATOR_H_

#include "korangar.h"

typedef enum {
	KR_ACTION_NONE,
	KR_ACTION_RENDER,
	KR_ACTION_CREATE,
	KR_ACTION_INIT,
	KR_ACTION_COMMIT,
	KR_ACTION_CAMERA_UPDATE,
	KR_ACTION_UPDATE,
	KR_ACTION_SETTINGS_UPDATE,
	KR_ACTION_DESTROY,
	KR_ACTION_ADS_ACTION,
	KR_ACTION_MAX
} kr_action;

typedef struct kr_integrator kr_integrator;

typedef kr_error(*kr_callback)(kr_integrator*, kr_handle, kr_action);

struct kr_integrator {
	kr_callback callback;
	kr_handle library;
	kr_handle context;
};

#ifdef __cplusplus
extern "C" {
#endif

kr_error kr_integrator_load(kr_integrator* integrator, const char* name);
kr_error kr_integrator_call(kr_integrator* integrator, kr_handle descriptor, kr_action action);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_INTEGRATOR_H_ */
