#ifndef _KORANGAR_INTERSECTORS_CUH_
#define _KORANGAR_INTERSECTORS_CUH_

#include "common/korangar.h"
#include "common/ads.h"
#include "common/scene.h"

#ifdef __cplusplus
extern "C" {
#endif


    kr_error kr_cuda_bvh_intersect(
        const kr_bvh_node* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count);

	kr_error kr_cuda_bvh_persistent_intersect(
        const kr_bvh_node_packed* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* rays,
        kr_intersection* isects, 
        u32 ray_count,
        u32 block_size,
        u32 grid_size,
        u32* warp_counter);

    kr_error kr_cuda_soa_bvh_persistent_intersect(
        const kr_SoA_BVH* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count,
        u32 block_size,
        u32 grid_size,
        u32* warp_counter);

    kr_error kr_cuda_obvh_intersect(
        const kr_bvh_node* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_mat4* transforms,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count);

    kr_error kr_cuda_obvh_persistent_intersect(
        const kr_bvh_node_packed* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_bvh_transformation_pair* transforms,
        const kr_ray* rays,
        kr_intersection* isects, 
        u32 ray_count,
        u32 block_size,
        u32 grid_size,
        u32* warp_counter);

    kr_error kr_cuda_obvh_compressed_persistent_intersect(
        const kr_SoA_OBVH_Compressed* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count,
        u32 block_size,
        u32 grid_size,
        u32* warp_counter);

    kr_error kr_cuda_soa_obvh_intersect(
        const kr_SoA_OBVH* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count,
        u32 block_size,
        u32 grid_size,
        u32* warp_counter);

    kr_error kr_cuda_bvh_intersect_bounds(
        const kr_bvh_node* bvh,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count, i32 max_level);

    kr_error kr_cuda_obvh_intersect_bounds(
        const kr_bvh_node* bvh,
        const kr_mat4* transforms,
        const kr_ray* rays,
        kr_intersection* isects,
        u32 ray_count, i32 max_level);

    kr_inline_device
        void bvh_cuda_persistent_intersect_ray(
            kr_handle tlas,
            const kr_ray* ray,
            kr_intersection* isect);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_INTERSECTORS_CUH_ */