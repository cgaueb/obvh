#ifndef _KORANGAR_INTERSECTORS_H_
#define _KORANGAR_INTERSECTORS_H_

#include "common/korangar.h"
#include "common/ads.h"
#include "common/scene.h"

#define KR_TRAVERSAL_METRICS 0
#define KR_PER_RAY_TRAVERSAL_METRICS 0

#ifdef __cplusplus
extern "C" {
#endif

	kr_error kr_bvh_intersect_ray(
		const kr_bvh_node* bvh, 
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* ray,
		kr_intersection* isect);

	kr_error kr_bvh_intersect(
        const kr_bvh_node* bvh, 
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_ray* rays,
        kr_intersection* isects, u32 ray_count);

    kr_error kr_obvh_intersect(
        const kr_bvh_node* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_mat43* transforms,
        const kr_ray* rays,
        kr_intersection* isects, u32 ray_count);

    kr_error kr_obvh_intersect_ray(
        const kr_bvh_node* bvh,
        const kr_vec3* vertices,
        const kr_uvec4* faces,
        const u32* primitives,
        const kr_mat43* transforms,
        const kr_ray* ray,
        kr_intersection* isect);

    kr_error kr_bvh_intersect_bounds_ray(
        const kr_bvh_node* bvh,
        const kr_ray* ray,
        kr_intersection* isect, i32 max_level);

    kr_error kr_bvh_intersect_bounds(
        const kr_bvh_node* bvh,
        const kr_ray* rays,
        kr_intersection* isects, u32 ray_count, i32 max_level);

    kr_error kr_obvh_intersect_bounds_ray(
        const kr_bvh_node* bvh,
        const kr_mat4* transforms,
        const kr_ray* ray,
        kr_intersection* isect, i32 max_level);

    kr_error kr_obvh_intersect_bounds(
        const kr_bvh_node* bvh,
        const kr_mat4* transforms,
        const kr_ray* rays,
        kr_intersection* isects, u32 ray_count, i32 max_level);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_INTERSECTORS_H_ */