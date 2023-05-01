#ifndef _KORANGAR_TRANSFORMS_H_
#define _KORANGAR_TRANSFORMS_H_

#include "common/korangar.h"
#include "common/ads.h"

#ifdef __cplusplus
extern "C" {
#endif

	kr_error kr_bvh_collapse(
        const kr_bvh_node* nodes,
        const u32* parents,
        kr_bvh_node* collapsed_nodes,
        u32* primitive_counts,
        u32* primitives,
        kr_scalar* costs,
        kr_scalar ci, kr_scalar ct,
        u32 leaf_count,
        u32 internal_count,
        const kr_obb3 * obbs = nullptr);

    kr_error kr_bvh_obb_tree(
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

    kr_scalar kr_bvh_sah(
        const kr_bvh_node* nodes, kr_i32 root_idx,
        kr_scalar ci, kr_scalar ct,
        const kr_obb3* obbs = nullptr,
        const char* outputFile = nullptr);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_TRANSFORMS_H_ */