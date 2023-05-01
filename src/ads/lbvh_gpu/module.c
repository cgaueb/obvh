#include "common/korangar.h"
#include "common/ads.h"

#include "lbvh.h"

KR_PUBLIC_API kr_error 
korangar_ads_action(kr_ads* ads, kr_handle descriptor, kr_ads_action action) {
    switch (action) {
    case KR_ADS_ACTION_CREATE:
        return (kr_ads*)lbvh_cuda_create();
        break;
    case KR_ADS_ACTION_INTERSECTION_QUERY:
        return lbvh_cuda_query_intersection((kr_ads_lbvh_cuda*)ads->context, (kr_intersection_query*)descriptor);
        break;
	case KR_ADS_ACTION_COMMIT:
		return (kr_error)lbvh_cuda_commit((kr_ads_lbvh_cuda*)ads->context, (kr_scene*)descriptor);
		break;
    case KR_ADS_ACTION_INIT:
        return lbvh_cuda_init((kr_ads_lbvh_cuda*)ads->context, (kr_descriptor_container*)descriptor);
        break;
	case KR_ADS_ACTION_SETTINGS_UPDATE:
		//return (kr_error)lbvh_cuda_settings_update((kr_ads_lbvh_cuda*)ads->context, (kr_descriptor_container*)descriptor);
		break;
    case KR_ADS_ACTION_CSV_EXPORT:
        //return lbvh_cuda_csv_export((kr_ads_lbvh_cuda*)ads->context, (kr_ads_csv_export*)descriptor);
        break;
    case KR_ADS_ACTION_DESTROY:
        //return lbvh_cuda_destroy((kr_ads_lbvh_cuda*)ads->context, (kr_scene*)descriptor);
        break;
    case KR_ADS_ACTION_NAME:
        return (kr_error)"LBVH GPU";
        break;
    case KR_ADS_ACTION_PREFERRED_QUERY_TYPE:
        return (kr_error)KR_QUERY_TYPE_CUDA;
        break;
    default:
        return kr_success;
        break;
    }

    return kr_success;
}