#ifndef _KORANGAR_ADS_H_
#define _KORANGAR_ADS_H_

#include "korangar.h"

typedef enum {
	KR_ADS_ACTION_NONE,
	KR_ADS_ACTION_INTERSECTION_QUERY,
	KR_ADS_ACTION_OCCLUSION_QUERY,
	KR_ADS_ACTION_PREFERRED_QUERY_TYPE,
	KR_ADS_ACTION_GET_BLAS,
	KR_ADS_ACTION_GET_TLAS,
	KR_ADS_ACTION_CREATE,
	KR_ADS_ACTION_COMMIT,
	KR_ADS_ACTION_INIT,
	KR_ADS_ACTION_UPDATE,
	KR_ADS_ACTION_SETTINGS_UPDATE,
	KR_ADS_ACTION_DESTROY,
	KR_ADS_ACTION_CSV_EXPORT,
	KR_ADS_ACTION_CSV_IMPORT,
	KR_ADS_ACTION_BVH_MODEL_EXPORT,
	KR_ADS_ACTION_BINARY_EXPORT,
	KR_ADS_ACTION_BINARY_IMPORT,
	KR_ADS_ACTION_NAME,
	KR_ADS_ACTION_MAX
} kr_ads_action;

typedef enum {
	KR_RUNTIME_TYPE_NONE,
	KR_RUNTIME_TYPE_CPU,
	KR_RUNTIME_TYPE_MAX
} kr_runtime_type;

typedef enum {
	KR_QUERY_TYPE_NONE,
	KR_QUERY_TYPE_CPU,
	KR_QUERY_TYPE_CUDA,
	KR_QUERY_TYPE_CUDA_MANAGED,
	KR_QUERY_TYPE_MAX
} kr_query_type;

typedef struct kr_ads kr_ads;

typedef kr_error(*kr_ads_callback)(kr_ads*, kr_handle, kr_ads_action);

typedef struct {
	kr_u32 core_count;
} kr_runtime_cpu;

typedef struct {
	kr_runtime_type type;
	union {
		kr_runtime_cpu cpu;
	};
} kr_runtime;

struct kr_ads {
	kr_ads_callback callback;
	kr_handle       library;
	kr_handle       context;
	kr_handle       runtime;
};

typedef struct {
	kr_vec3 origin;
	kr_scalar tmin;
	kr_vec3 direction;
	kr_scalar tmax;
} kr_ray;

typedef struct {
	kr_vec3 geom_normal;
	kr_vec2 barys;
	kr_u32  primitive;
	kr_u32  instance;
} kr_intersection;

typedef kr_i32 kr_occlusion;

typedef struct {
	const kr_ray* rays;
	kr_occlusion* isects;
} kr_occlusion_query_cpu;

typedef struct {
	kr_query_type type;
	union {
		kr_occlusion_query_cpu cpu;
	};
} kr_occlusion_query;

typedef struct {
	const kr_ray* rays;
	kr_intersection* isects;
	kr_size ray_count;
} kr_intersection_query_cpu;


typedef struct {
	const kr_ray* rays;
	kr_intersection* isects;
	kr_size ray_count;
} kr_intersection_query_cuda;

typedef kr_intersection_query_cuda kr_intersection_query_cuda_managed;

typedef struct {
	kr_query_type type;
	union {
		kr_intersection_query_cpu cpu;
		kr_intersection_query_cuda cuda;
		kr_intersection_query_cuda_managed cuda_managed;
	};
} kr_intersection_query;

typedef struct {
	const char* name;
	const char* path;
} kr_ads_csv_export;

typedef struct {
	const char* name;
	const char* path;
} kr_ads_bvh_model_export;

typedef struct {
	kr_ads_action action;
	kr_handle     data;
} kr_ads_action_payload;

typedef struct {
    aabb3 bounds;   
    union {
        u32 primitivesOffset;    // leaf
        struct {
            u32 left;   // interior
            u32 right;   // interior
        };
    };
    u16 nPrimitives;  // 0 -> interior node
    u8 axis;          // interior node: xyz
} kr_bvh_node;

typedef struct {
	union {
		struct {
			aabb3 lbounds;
			aabb3 rbounds;
		};
		struct {
			vec4 n0xy;
			vec4 n1xy;
			vec4 nz;
		};
	};
	union {
		u32 primitivesOffset;    // leaf
		struct {
			u32 left;   // interior
			u32 right;   // interior
		};
		ivec4 metadata;
	};
} kr_bvh_node_packed;

typedef struct {
	mat43* left_T;
	mat43* right_T;
	ivec4* metadata;
} kr_SoA_OBVH;

typedef struct {
	aabb3* lbounds;
	aabb3* rbounds;
	ivec4* metadata;
} kr_SoA_OBVH_Compressed;

typedef struct {
	vec4* lbbox_XY;
	vec4* rbbox_XY;
	vec4* lrbbox_Z;
	ivec4* metadata;
} kr_SoA_BVH;

typedef struct {
	kr_mat4 l;
	kr_mat4 r;
	//kr_mat43 l;
	//kr_mat43 r;
} kr_bvh_transformation_pair;

typedef struct {
    aabb3 bounds;
    union {
        u32 primitivesOffset;    // leaf
        u32 secondChildOffset;   // interior
    };
    u16 nPrimitives;  // 0 -> interior node
    u8 axis;          // interior node: xyz
    u8 pad[1];        // ensure 32 byte total size
} kr_linear_bvh_node;

typedef struct {
	const kr_vec3* vertices;
	const kr_uvec4* faces;
	const u32* primitives;
	kr_handle* bvh;
} kr_cuda_blas;

typedef struct
{
	kr_scalar collapse_mark_time;
	kr_scalar collapse_time;
}kr_ads_collapse_metrics;

typedef struct
{
	kr_scalar obb_projection_time;
	kr_scalar obb_candidates_eval_time;
	kr_scalar obb_refit_time;
	kr_scalar obb_finalize_time;
}kr_ads_obb_metrics;

//#define KR_MORTON_64
#ifdef KR_MORTON_64
typedef kr_u64 kr_morton;
#else
typedef kr_u32 kr_morton;
#endif

#ifdef __cplusplus
extern "C" {
#endif

kr_error kr_ads_load(kr_ads* ads, const char* name);
kr_error kr_ads_call(kr_ads* ads, kr_handle descriptor, kr_ads_action action);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_ADS_H_ */
