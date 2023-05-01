#include "common/korangar.h"
#include "common/integrator.h"

#include "bench.h"

KR_PUBLIC_API kr_error 
korangar_action(kr_integrator* integrator, kr_handle descriptor, kr_action action) {
	switch (action) {
    case KR_ACTION_CREATE:
        return (kr_error) integrator_bench_create();
        break;
    case KR_ACTION_RENDER:
        return integrator_bench_render((kr_integrator_bench*)integrator->context, (kr_target*)descriptor);
        break;
    case KR_ACTION_COMMIT:
        return integrator_bench_commit((kr_integrator_bench*)integrator->context, (kr_scene*)descriptor);
        break;
    case KR_ACTION_INIT:
        return integrator_bench_init((kr_integrator_bench*)integrator->context, (kr_descriptor_container*)descriptor);
        break;
    case KR_ACTION_ADS_ACTION: {
        kr_ads_action_payload* payload = (kr_ads_action_payload*)descriptor;
        kr_integrator_bench* bench = (kr_integrator_bench*)integrator->context;
        kr_ads_call(bench->ads, payload->data, payload->action);
    } 
        break;
    case KR_ACTION_UPDATE:
        break;
    case KR_ACTION_CAMERA_UPDATE:
        break;
    default:
        return kr_success;
        break;
	}
    return kr_success;
}
