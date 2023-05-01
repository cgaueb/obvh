#ifndef _KORANGAR_TRANSFORMS_CUH_
#define _KORANGAR_TRANSFORMS_CUH_

#include "common/korangar.h"
#include "common/ads.h"
#include "common/scene.h"

#ifdef __cplusplus
extern "C" {
#endif

    kr_error kr_cuda_bbox_calculate(
        cuvec4* faces,
        cvec3* vertices,
        u32 face_count,
        kr_aabb3* aabb
    );

    kr_error kr_cuda_centroid_bbox_calculate(
        cuvec4* faces,
        cvec3* vertices,
        u32 face_count,
        kr_aabb3* aabb
    );

    kr_error kr_cuda_bvh_persistent(
        const kr_bvh_node* nodes,
        kr_bvh_node_packed* packed_nodes,
        u32 node_count
    );

    kr_error kr_cuda_obvh_persistent(
        const kr_bvh_node* nodes,
        const kr_mat4* transforms,
        kr_bvh_node_packed* packed_nodes,
        kr_bvh_transformation_pair* packed_transforms,
        u32 node_count
    );

	kr_error kr_cuda_bvh_collapse(
        const kr_bvh_node* nodes,
        const u32* parents,
        kr_bvh_node* collapsed_nodes,
        u32* primitive_counts,
        u32* primitives,
        kr_scalar* costs,
        kr_scalar ci, kr_scalar ct,
        u32 leaf_count,
        u32 internal_count,
        kr_ads_collapse_metrics * metrics);

    kr_error kr_cuda_bvh_obb_tree(
        kr_bvh_node* nodes,
        const u32* parents,
        const u32* primitive_counts,
        const kr_scalar* costs,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        kr_obb3* obbs,
        kr_mat4* transforms,
        kr_scalar ci, kr_scalar ct, kr_scalar obb_cost,
        u32 primitive_count,
        u32 internal_count,
        u32 primitive_threshold);

    kr_error kr_cuda_soa_bvh(
        const kr_bvh_node_packed* packed_bvh,
        kr_SoA_BVH*& soa_bvh,
        u32 node_count);

    kr_error kr_cuda_soa_obvh(
        const kr_bvh_node_packed* packed_bvh,
        const kr_bvh_transformation_pair* packed_transforms,
        kr_SoA_OBVH*& soa_bvh,
        u32 internal_count,
        u32 node_count);

    kr_error kr_cuda_soa_compressed_obvh(
        const kr_bvh_node_packed* packed_bvh,
        const kr_bvh_transformation_pair* packed_transforms,
        kr_SoA_OBVH_Compressed*& soa_bvh,
        u32 internal_count,
        u32 node_count);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_TRANSFORMS_CUH_ */