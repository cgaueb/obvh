#ifndef _KORANGAR_LBVH_CUDA_H_
#define _KORANGAR_LBVH_CUDA_H_

#include "common/cuda/scene.h"
#include "common/ads.h"

typedef kr_aabb3 kr_minmax_pair;

typedef struct {
    aabb3 aabb;
    aabb3 centroid_aabb;
    
    kr_object_cu* d_object_cu;
    kr_object_cu* h_object_cu;

    kr_bvh_node* bvh;
    kr_bvh_node* collapsed_bvh;
	kr_linear_bvh_node* linear_bvh;
    kr_bvh_node_packed* bvh_packed;
    kr_SoA_OBVH_Compressed* soa_obvh_compressed;
    kr_SoA_OBVH* soa_obvh_packed;
    kr_SoA_BVH* soa_bvh_packed;

    kr_mat4* obb_transforms;
    kr_bvh_transformation_pair* transformations;

    kr_morton* mortons;
    kr_scalar* min_proj;
    kr_scalar* max_proj;
    kr_minmax_pair* minmax_pairs;
    kr_obb3* obbs;
    u32* primitive_counts;
    u32* parents;
    u32* primitives;
    b32* visit_table;
    b32* collapse_table;
    kr_scalar* costs;
    u32* ray_counter;

    u32 node_count;
    u32 internal_count;
    u32 primitive_count;
    u32 leaf_count;
} kr_ads_blas_cuda;

typedef struct {
    kr_ads_blas_cuda blas;
    
    aabb3 aabb;
    aabb3 centroid_aabb;

    kr_scene* scene;
    kr_scene_cu* scene_cu;
        
    kr_scalar cost_traversal;
    kr_scalar cost_internal;

    i32 intersect_bounds_level;

    u32 version;
    u32 cu_core_count;
    u32 cu_blockSize;
    u32 cu_gridSize;

    b32 save_construction_information;
    b32 collapse;
    b32 use_obbs;
    b32 intersect_bounds;
    b32 use_persistent_kernel;

} kr_ads_bvh2_cuda;

#ifdef __cplusplus
extern "C" {
#endif

kr_error atrbvh_cuda_query_intersection(kr_ads_bvh2_cuda* ads, kr_intersection_query* query);
kr_error atrbvh_cuda_commit(kr_ads_bvh2_cuda* ads, kr_scene* scene);
kr_error atrbvh_cuda_init(kr_ads_bvh2_cuda* ads, kr_descriptor_container* settings);
kr_ads_bvh2_cuda* atrbvh_cuda_create();

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_LBVH_CUDA_H_ */