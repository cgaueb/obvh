#define KR_VECMATH_IMPL
#include "common/vecmath.h"
#include "common/logger.h"
#include "common/util.h"

#include "ads/common/cuda/intersectors.cuh"
#include "common/cuda/util.h"
#include "common/cuda/util.cuh"

#include <stdio.h>

#define KR_PER_RAY_TRAVERSAL_METRICS 1

kr_internal kr_global
void bvh_intersect(
	const kr_bvh_node* __restrict__ bvh,
	const kr_vec3* __restrict__ vertices,
	const kr_uvec4* __restrict__ faces,
	const u32* __restrict__ primitives,
	const kr_ray* __restrict__ rays,
	kr_intersection* isects,
	u32 ray_count) {
	const u32 ray_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_index >= ray_count)
		return;

	const kr_bvh_node* nodes = bvh;
	const kr_ray* ray = rays + ray_index;
	kr_intersection* isect = isects + ray_index;
	kr_scalar min_distance = ray->tmax;
	isect->primitive = kr_invalid_index;

	cvec3 origin = ray->origin;
	cvec3 direction = ray->direction;
	cvec3 idirection = kr_vinverse3(ray->direction);
	vec2 barys = { 0, 0 };
	u32 hit_index = 0xFFFFFFFF;

	constexpr kr_aabb3 unit_cube = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;

	i32 toVisitOffset = 0, currentNodeIndex = 0;
	i32 nodesToVisit[KR_LBVH_CUDA_STACK_SIZE] = { 0 };
	while (kr_true) {
		const kr_bvh_node* node = &nodes[currentNodeIndex];

		if (node->nPrimitives > 0) {
			for (int i = 0; i < node->nPrimitives; ++i) {
				u32 primitive_id = (node->nPrimitives == 1) ? node->primitivesOffset : primitives[node->primitivesOffset + i];

				cuvec4 face = faces[primitive_id];
				cvec3  va = vertices[face.x];
				cvec3  vb = vertices[face.y];
				cvec3  vc = vertices[face.z];

				vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, min_distance);
				if (tisect.z < min_distance) {
					min_distance = tisect.z;
					hit_index = (u32)primitive_id;
					barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
				}
			}
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
		else {
			const kr_bvh_node* l = &nodes[node->left];
			const kr_bvh_node* r = &nodes[node->right];

			cvec2 lisects = kr_ray_aabb_intersect(direction, origin, l->bounds, min_distance);
			cvec2 risects = kr_ray_aabb_intersect(direction, origin, r->bounds, min_distance);

			b32 traverseChild0 = (lisects.y >= lisects.x);
			b32 traverseChild1 = (risects.y >= risects.x);
			if (traverseChild0 != traverseChild1) {
				currentNodeIndex = (traverseChild0) ? node->left : node->right;
			}
			else {
				if (!traverseChild0) {
					if (toVisitOffset == 0) break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
					continue;
				}
				else {
					if (lisects.x < risects.x) {
						nodesToVisit[toVisitOffset++] = node->right;
						currentNodeIndex = node->left;
					}
					else {
						nodesToVisit[toVisitOffset++] = node->left;
						currentNodeIndex = node->right;
					}
				}
			}
		}
	}

	if (hit_index == 0xFFFFFFFF) {
		return;
	}

	cuvec4 face = faces[hit_index];
	cvec3  va = vertices[face.x];
	cvec3  vb = vertices[face.y];
	cvec3  vc = vertices[face.z];

	isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
	isect->primitive = hit_index;
	isect->barys = { barys.x, barys.y };
}


#define WARP_FULL_MASK 0xffffffff
kr_internal kr_global
void bvh_persistent_intersect(
    const kr_bvh_node_packed* bvh, 
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    const u32* primitives,
    const kr_ray* rays, 
    kr_intersection* isects, 
    u32 ray_count,
    u32* warp_counter)
{
	const kr_bvh_node_packed* nodes = bvh;
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool more_work = true;

	vec3 origin;
	vec3 direction;
	vec3 normal;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr kr_aabb3 unit_cube = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	//constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0xFFFFFFFF;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x76543210;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;

	while (__ballot_sync(WARP_FULL_MASK, more_work == true)) {
		ray_index = atomicAdd(warp_counter, 1);
		more_work = !(ray_index >= ray_count);
		const kr_ray* ray = kr_null;
		kr_intersection* isect = kr_null;

		internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
		leafNodeIdx = 0;
		hit_index = 0xFFFFFFFF;
		stackPtr = 0;
#if KR_PER_RAY_TRAVERSAL_METRICS
		i32 internal_count = 0;
		i32 leaf_count = 0;
#endif
		if (more_work) {
			ray = &rays[ray_index];

			isect = &isects[ray_index];
			isect->primitive = hit_index;
			isect->barys = { 0.5, 0.5 };
			isect->geom_normal = ray->direction;

			origin = ray->origin;
			direction = ray->direction;
			tmax = ray->tmax;
#if KR_PER_RAY_TRAVERSAL_METRICS
			isect->instance = 0;
#endif
			hitT = tmax;
			internalNodeIdx = 0;
		}

		int active_rays = __ballot_sync(WARP_FULL_MASK, more_work == true);
		while (internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
			bool searchingLeaf = true;

			while (searchingLeaf && internalNodeIdx >= 0 && internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
				const kr_aabb3& lbounds = nodes[internalNodeIdx].lbounds;
				const kr_aabb3& rbounds = nodes[internalNodeIdx].rbounds;
				const ivec4& metadata = nodes[internalNodeIdx].metadata;
#if KR_PER_RAY_TRAVERSAL_METRICS
				internal_count++;
#endif
				cvec2 risects = kr_ray_aabb_intersect(direction, origin, rbounds, hitT);
				cvec2 lisects = kr_ray_aabb_intersect(direction, origin, lbounds, hitT);

				bool traverseLeft = (lisects.y >= lisects.x) && (lisects.y >= 0.0) && (lisects.x <= hitT);
				bool traverseRight = (risects.y >= risects.x) && (risects.y >= 0.0) && (risects.x <= hitT);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				if (traverseLeft != traverseRight) {
					if (traverseRight) internalNodeIdx = right_internalNodeIdx;
				}
				else {
					if (!traverseLeft) {
						internalNodeIdx = traversalStack[stackPtr--];
					}
					else {
						if (risects.x < lisects.x) {
							// Swap node IDs
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
					searchingLeaf = false;
					leafNodeIdx = internalNodeIdx;
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			__syncwarp(active_rays);

			while (leafNodeIdx < 0) {
				const ivec4& metadata = nodes[~leafNodeIdx].metadata;
				i32 beg = metadata.x;
				i32 end = metadata.y;
				i32 num = end - beg;
#if KR_PER_RAY_TRAVERSAL_METRICS
				leaf_count += num;
#endif
				for (; beg < end; ++beg) {
					u32 primitive_id = (num == 1) ? beg : primitives[beg];
					
					cuvec4 face = faces[primitive_id];
					cvec3  va = vertices[face.x];
					cvec3  vb = vertices[face.y];
					cvec3  vc = vertices[face.z];

					vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
					if (tisect.z < hitT) {
						hitT = tisect.z;
						hit_index = (u32)primitive_id;
						barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
					}
				}
				leafNodeIdx = internalNodeIdx;
				if (internalNodeIdx < 0)
				{
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			active_rays = __ballot_sync(active_rays, internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL);
		}

#if KR_PER_RAY_TRAVERSAL_METRICS
		if(isect) isect->instance = (leaf_count << 16) | internal_count;
#else
		if(isect) isect->instance = 0;
#endif

		if (hit_index == 0xFFFFFFFF)
			continue;

		cuvec4 face = faces[hit_index];
		cvec3  va = vertices[face.x];
		cvec3  vb = vertices[face.y];
		cvec3  vc = vertices[face.z];

		isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
		isect->primitive = hit_index;
		isect->barys = { barys.x, barys.y };
	}
}

kr_internal kr_global
void soa_bvh_persistent_intersect(
	const kr_SoA_BVH* bvh,
	const kr_vec3* vertices,
	const kr_uvec4* faces,
	const u32* primitives,
	const kr_ray* rays,
	kr_intersection* isects,
	u32 ray_count,
	u32* warp_counter)
{
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool more_work = true;

	vec3 origin;
	vec3 direction;
	vec3 normal;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr kr_aabb3 unit_cube = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	//constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0xFFFFFFFF;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x76543210;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;

	while (__ballot_sync(WARP_FULL_MASK, more_work == true)) {
		ray_index = atomicAdd(warp_counter, 1);
		more_work = !(ray_index >= ray_count);
		const kr_ray* ray = kr_null;
		kr_intersection* isect = kr_null;

		internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
		leafNodeIdx = 0;
		hit_index = 0xFFFFFFFF;
		stackPtr = 0;

		if (more_work) {
			ray = &rays[ray_index];
			isect = &isects[ray_index];
			isect->barys = { 0.5, 0.5 };
			isect->geom_normal = ray->direction;

			origin = ray->origin;
			direction = ray->direction;
			tmax = ray->tmax;
			hitT = tmax;
			internalNodeIdx = 0;
		}

		int active_rays = __ballot_sync(WARP_FULL_MASK, more_work == true);
		while (internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
			bool searchingLeaf = true;

			while (searchingLeaf && internalNodeIdx >= 0 && internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
				const vec4& lbounds_XY = bvh->lbbox_XY[internalNodeIdx];
				const vec4& rbounds_XY = bvh->rbbox_XY[internalNodeIdx];
				const vec4& lrbbox_Z = bvh->lrbbox_Z[internalNodeIdx];

				kr_aabb3 lbounds;
				kr_aabb3 rbounds;

				lbounds.min.x = lbounds_XY.x;
				lbounds.min.y = lbounds_XY.z;
				lbounds.min.z = lrbbox_Z.x;

				lbounds.max.x = lbounds_XY.y;
				lbounds.max.y = lbounds_XY.w;
				lbounds.max.z = lrbbox_Z.y;

				rbounds.min.x = rbounds_XY.x;
				rbounds.min.y = rbounds_XY.z;
				rbounds.min.z = lrbbox_Z.z;

				rbounds.max.x = rbounds_XY.y;
				rbounds.max.y = rbounds_XY.w;
				rbounds.max.z = lrbbox_Z.w;

				const ivec4& metadata = bvh->metadata[internalNodeIdx];

				cvec2 risects = kr_ray_aabb_intersect(direction, origin, rbounds, hitT);
				cvec2 lisects = kr_ray_aabb_intersect(direction, origin, lbounds, hitT);

				bool traverseLeft = (lisects.y >= lisects.x);
				bool traverseRight = (risects.y >= risects.x);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				if (traverseLeft != traverseRight) {
					if (traverseRight) internalNodeIdx = right_internalNodeIdx;
				}
				else {
					if (!traverseLeft) {
						internalNodeIdx = traversalStack[stackPtr--];
					}
					else {
						if (risects.x < lisects.x) {
							// Swap node IDs
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
					searchingLeaf = false;
					leafNodeIdx = internalNodeIdx;
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			__syncwarp(active_rays);

			while (leafNodeIdx < 0) {
				const ivec4& metadata = bvh->metadata[~leafNodeIdx];
				i32 beg = metadata.x;
				i32 end = metadata.y;
				i32 num = end - beg;
				for (; beg < end; ++beg) {
					u32 primitive_id = (num == 1) ? beg : primitives[beg];

					cuvec4 face = faces[primitive_id];
					cvec3  va = vertices[face.x];
					cvec3  vb = vertices[face.y];
					cvec3  vc = vertices[face.z];

					vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
					if (tisect.z < hitT) {
						hitT = tisect.z;
						hit_index = (u32)primitive_id;
						barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
					}
				}
				leafNodeIdx = internalNodeIdx;
				if (internalNodeIdx < 0)
				{
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			active_rays = __ballot_sync(active_rays, internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL);
		}

		if (hit_index == 0xFFFFFFFF)
			continue;

		cuvec4 face = faces[hit_index];
		cvec3  va = vertices[face.x];
		cvec3  vb = vertices[face.y];
		cvec3  vc = vertices[face.z];

		isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
		isect->primitive = hit_index;
		isect->barys = { barys.x, barys.y };
	}
}

kr_error kr_cuda_bvh_intersect(
	const kr_bvh_node* bvh,
	const kr_vec3* vertices,
	const kr_uvec4* faces,
	const u32* primitives,
	const kr_ray* rays,
	kr_intersection* isects,
	u32 ray_count) {

	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);

	bvh_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives,
		rays, isects, ray_count);

	return kr_success;
}

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
	u32* warp_counter) {

	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);
	//dim3 blockSize = dim3(block_size);
	//dim3 gridSize = dim3(grid_size);
	bvh_persistent_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives,
		rays, isects, ray_count, warp_counter);

	return kr_success;
}

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
	u32* warp_counter)
{
	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);
	//dim3 blockSize = dim3(block_size);
	//dim3 gridSize = dim3(grid_size);
	soa_bvh_persistent_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives,
		rays, isects, ray_count, warp_counter);

	return kr_success;
}

kr_internal kr_global
void obvh_intersect(
	const kr_bvh_node* __restrict__ bvh,
	const kr_vec3* __restrict__ vertices,
	const kr_uvec4* __restrict__ faces,
	const u32* __restrict__ primitives,
	const kr_mat4* __restrict__ transforms,
	const kr_ray* __restrict__ rays,
	kr_intersection* isects,
	u32 ray_count) {
	const u32 ray_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_index >= ray_count)
		return;

	const kr_bvh_node* nodes = bvh;
	const kr_ray* ray = rays + ray_index;
	kr_intersection* isect = isects + ray_index;
	kr_scalar min_distance = ray->tmax;
	isect->primitive = kr_invalid_index;

	cvec3 origin = ray->origin;
	cvec3 direction = ray->direction;
	cvec3 idirection = kr_vinverse3(ray->direction);
	vec2 barys = { 0, 0 };
	u32 hit_index = 0xFFFFFFFF;

	constexpr kr_aabb3 unit_cube = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;

	i32 toVisitOffset = 0, currentNodeIndex = 0;
	i32 nodesToVisit[KR_LBVH_CUDA_STACK_SIZE] = { 0 };
	while (kr_true) {
		const kr_bvh_node* node = &nodes[currentNodeIndex];

		if (node->nPrimitives > 0) {
			for (int i = 0; i < node->nPrimitives; ++i) {
				u32 primitive_id = (node->nPrimitives == 1) ? node->primitivesOffset : primitives[node->primitivesOffset + i];

				cuvec4 face = faces[primitive_id];
				cvec3  va = vertices[face.x];
				cvec3  vb = vertices[face.y];
				cvec3  vc = vertices[face.z];

				vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, min_distance);
				if (tisect.z < min_distance) {
					min_distance = tisect.z;
					hit_index = (u32)primitive_id;
					barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
				}
			}
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
		else {
			const kr_bvh_node* l = &nodes[node->left];
			const kr_bvh_node* r = &nodes[node->right];

			vec2 lisects = { 0 };
			vec2 risects = { 0 };

			if (l->axis == 1) {
				const kr_mat4* ltransform = &transforms[node->left];
				cvec3 lorigin_lcs = kr_vtransform3p(ltransform, &origin);
				cvec3 ldirection_lcs = kr_ntransform3p(ltransform, &direction);
				lisects = kr_ray_unit_aabb_intersect(ldirection_lcs, lorigin_lcs, min_distance);
			}
			else {
				lisects = kr_ray_aabb_intersect(direction, origin, l->bounds, min_distance);
			}

			if (r->axis == 1) {
				const kr_mat4* rtransform = &transforms[node->right];
				cvec3 rorigin_lcs = kr_vtransform3p(rtransform, &origin);
				cvec3 rdirection_lcs = kr_ntransform3p(rtransform, &direction);
				risects = kr_ray_unit_aabb_intersect(rdirection_lcs, rorigin_lcs, min_distance);
			}
			else {
				risects = kr_ray_aabb_intersect(direction, origin, r->bounds, min_distance);
			}

			b32 traverseChild0 = (lisects.y >= lisects.x);
			b32 traverseChild1 = (risects.y >= risects.x);
			if (traverseChild0 != traverseChild1) {
				currentNodeIndex = (traverseChild0) ? node->left : node->right;
			}
			else {
				if (!traverseChild0) {
					if (toVisitOffset == 0) break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
					continue;
				}
				else {
					if (lisects.x < risects.x) {
						nodesToVisit[toVisitOffset++] = node->right;
						currentNodeIndex = node->left;
					}
					else {
						nodesToVisit[toVisitOffset++] = node->left;
						currentNodeIndex = node->right;
					}
				}
			}
		}
	}

	if (hit_index == 0xFFFFFFFF) {
		return;
	}

	cuvec4 face = faces[hit_index];
	cvec3  va = vertices[face.x];
	cvec3  vb = vertices[face.y];
	cvec3  vc = vertices[face.z];

	isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
	isect->primitive = hit_index;
	isect->barys = { barys.x, barys.y };
}

kr_internal kr_global
void obvh_persistent_intersect(
	const kr_bvh_node_packed* __restrict__ bvh,
	const kr_vec3* __restrict__ vertices,
	const kr_uvec4* __restrict__ faces,
	const u32* __restrict__ primitives,
	const kr_bvh_transformation_pair* __restrict__ transforms,
	const kr_ray* __restrict__ rays,
	kr_intersection* isects,
	u32 ray_count,
	u32* warp_counter) {
	const kr_bvh_node_packed* nodes = bvh;
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool more_work = true;

	vec3 origin;
	vec3 direction;
	vec3 normal;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr kr_aabb3 unit_cube = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	//constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0xFFFFFFFF;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x76543210;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;

	while (__ballot_sync(WARP_FULL_MASK, more_work == true)) {
		ray_index = atomicAdd(warp_counter, 1);
		more_work = !(ray_index >= ray_count);
		const kr_ray* ray = kr_null;
		kr_intersection* isect = kr_null;

		internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
		leafNodeIdx = 0;
		hit_index = 0xFFFFFFFF;
		stackPtr = 0;

		if (more_work) {
			ray = &rays[ray_index];
			isect = &isects[ray_index];
			isect->barys = { 0.5, 0.5 };
			isect->geom_normal = ray->direction;

			origin = ray->origin;
			direction = ray->direction;
			tmax = ray->tmax;
			hitT = tmax;
			internalNodeIdx = 0;
		}

		int active_rays = __ballot_sync(WARP_FULL_MASK, more_work == true);
		while (internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
			bool searchingLeaf = true;

			while (searchingLeaf && internalNodeIdx >= 0 && internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
				const ivec4& metadata = nodes[internalNodeIdx].metadata;

				/*cvec2 lisects = (metadata.z == 1)
					? kr_ray_aabb_intersect(kr_ntransform3p(&transforms[internalNodeIdx].l, &direction), kr_vtransform3p(&transforms[internalNodeIdx].l, &origin), unit_cube, hitT)
					: kr_ray_aabb_intersect(direction, origin, nodes[internalNodeIdx].lbounds, hitT);
				cvec2 risects = (metadata.z == 1)
					? kr_ray_aabb_intersect(kr_ntransform3p(&transforms[internalNodeIdx].r, &direction), kr_vtransform3p(&transforms[internalNodeIdx].r, &origin), unit_cube, hitT)
					: kr_ray_aabb_intersect(direction, origin, nodes[internalNodeIdx].rbounds, hitT);
				*/

				cvec2 lisects = kr_ray_unit_aabb_intersect(kr_ntransform3p(&transforms[internalNodeIdx].l, &direction), kr_vtransform3p(&transforms[internalNodeIdx].l, &origin), hitT);
				cvec2 risects = kr_ray_unit_aabb_intersect(kr_ntransform3p(&transforms[internalNodeIdx].r, &direction), kr_vtransform3p(&transforms[internalNodeIdx].r, &origin), hitT);
				//cvec2 lisects = kr_ray_aabb_intersect(kr_n43transform3p(&transforms[internalNodeIdx].l, &direction), kr_v43transform3p(&transforms[internalNodeIdx].l, &origin), unit_cube, hitT);
				//cvec2 risects = kr_ray_aabb_intersect(kr_n43transform3p(&transforms[internalNodeIdx].r, &direction), kr_v43transform3p(&transforms[internalNodeIdx].r, &origin), unit_cube, hitT);

				bool traverseLeft = (lisects.y >= lisects.x);
				bool traverseRight = (risects.y >= risects.x);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				if (traverseLeft != traverseRight) {
					if (traverseRight) internalNodeIdx = right_internalNodeIdx;
				}
				else {
					if (!traverseLeft) {
						internalNodeIdx = traversalStack[stackPtr--];
					}
					else {
						if (risects.x < lisects.x) {
							// Swap node IDs
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
					searchingLeaf = false;
					leafNodeIdx = internalNodeIdx;
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			__syncwarp(active_rays);

			while (leafNodeIdx < 0) {
				const ivec4& metadata = nodes[~leafNodeIdx].metadata;
				i32 beg = metadata.x;
				i32 end = metadata.y;
				i32 num = end - beg;
				for (; beg < end; ++beg) {
					u32 primitive_id = (num == 1) ? beg : primitives[beg];

					cuvec4 face = faces[primitive_id];
					cvec3  va = vertices[face.x];
					cvec3  vb = vertices[face.y];
					cvec3  vc = vertices[face.z];

					vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
					if (tisect.z < hitT) {
						hitT = tisect.z;
						hit_index = (u32)primitive_id;
						barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
					}
				}
				leafNodeIdx = internalNodeIdx;
				if (internalNodeIdx < 0)
				{
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			active_rays = __ballot_sync(active_rays, internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL);
		}

		if (hit_index == 0xFFFFFFFF)
			continue;

		cuvec4 face = faces[hit_index];
		cvec3  va = vertices[face.x];
		cvec3  vb = vertices[face.y];
		cvec3  vc = vertices[face.z];

		isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
		isect->primitive = hit_index;
		isect->barys = { barys.x, barys.y };
	}
}

kr_internal kr_global
void soa_obvh_persistent_intersect(
	const kr_SoA_OBVH* __restrict__ bvh,
	const kr_vec3* __restrict__ vertices,
	const kr_uvec4* __restrict__ faces,
	const u32* __restrict__ primitives,
	const kr_ray* __restrict__ rays,
	kr_intersection* isects,
	u32 ray_count,
	u32* warp_counter) {
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool more_work = true;

	vec3 origin;
	vec3 direction;
	vec3 normal;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x76543210;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;

	while (__ballot_sync(WARP_FULL_MASK, more_work == true)) {
		ray_index = atomicAdd(warp_counter, 1);
		more_work = !(ray_index >= ray_count);
		const kr_ray* ray = kr_null;
		kr_intersection* isect = kr_null;

		internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
		leafNodeIdx = 0;
		hit_index = 0xFFFFFFFF;
		stackPtr = 0;
#if KR_PER_RAY_TRAVERSAL_METRICS
		i32 internal_count = 0;
		i32 leaf_count = 0;
#endif
		if (more_work) {
			ray = &rays[ray_index];
			isect = &isects[ray_index];
			isect->primitive = hit_index;
			isect->barys = { 0.5, 0.5 };
			isect->geom_normal = ray->direction;

			origin = ray->origin;
			direction = ray->direction;
			tmax = ray->tmax;
#if KR_PER_RAY_TRAVERSAL_METRICS
			isect->instance = 0;
#endif
			hitT = tmax;
			internalNodeIdx = 0;
		}

		int active_rays = __ballot_sync(WARP_FULL_MASK, more_work == true);
		while (internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
			bool searchingLeaf = true;

			while (searchingLeaf && internalNodeIdx >= 0 && internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
				const ivec4& metadata = bvh->metadata[internalNodeIdx];
				const mat43* left_T = &bvh->left_T[internalNodeIdx];
				const mat43* right_T = &bvh->right_T[internalNodeIdx];
#if KR_PER_RAY_TRAVERSAL_METRICS
				internal_count++;
#endif
				/*cvec2 lisects = (metadata.z == 1)
					? kr_ray_aabb_intersect(kr_ntransform3p(&transforms[internalNodeIdx].l, &direction), kr_vtransform3p(&transforms[internalNodeIdx].l, &origin), unit_cube, hitT)
					: kr_ray_aabb_intersect(direction, origin, nodes[internalNodeIdx].lbounds, hitT);
				cvec2 risects = (metadata.w == 1)
					? kr_ray_aabb_intersect(kr_ntransform3p(&transforms[internalNodeIdx].r, &direction), kr_vtransform3p(&transforms[internalNodeIdx].r, &origin), unit_cube, hitT)
					: kr_ray_aabb_intersect(direction, origin, nodes[internalNodeIdx].rbounds, hitT);*/
				
				cvec2 lisects = kr_ray_unit_aabb_intersect(kr_n43transform3p(left_T, &direction), kr_v43transform3p(left_T, &origin), hitT);
				cvec2 risects = kr_ray_unit_aabb_intersect(kr_n43transform3p(right_T, &direction), kr_v43transform3p(right_T, &origin), hitT);

				const bool traverseLeft = (lisects.y >= lisects.x) && (lisects.y >= 0.0) && (lisects.x <= hitT);
				const bool traverseRight = (risects.y >= risects.x) && (risects.y >= 0.0) && (risects.x <= hitT);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				if (traverseLeft != traverseRight) {
					if (traverseRight) internalNodeIdx = right_internalNodeIdx;
				}
				else {
					if (!traverseLeft) {
						internalNodeIdx = traversalStack[stackPtr--];
					}
					else {
						if (risects.x < lisects.x) {
							// Swap node IDs
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
					searchingLeaf = false;
					leafNodeIdx = internalNodeIdx;
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			__syncwarp(active_rays);

			while (leafNodeIdx < 0) {
				const ivec4& metadata = bvh->metadata[~leafNodeIdx];
				i32 beg = metadata.x;
				i32 end = metadata.y;
				i32 num = end - beg;
#if KR_PER_RAY_TRAVERSAL_METRICS
				leaf_count += num;
#endif
				for (; beg < end; ++beg) {
					u32 primitive_id = (num == 1) ? beg : primitives[beg];

					cuvec4 face = faces[primitive_id];
					cvec3  va = vertices[face.x];
					cvec3  vb = vertices[face.y];
					cvec3  vc = vertices[face.z];

					vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
					if (tisect.z < hitT) {
						hitT = tisect.z;
						hit_index = (u32)primitive_id;
						barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
					}
				}
				leafNodeIdx = internalNodeIdx;
				if (internalNodeIdx < 0)
				{
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			active_rays = __ballot_sync(active_rays, internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL);
		}

#if KR_PER_RAY_TRAVERSAL_METRICS
		if (isect) isect->instance = (leaf_count << 16) | internal_count;
#else
		if (isect) isect->instance = 0;
#endif

		if (hit_index == 0xFFFFFFFF)
			continue;

		cuvec4 face = faces[hit_index];
		cvec3  va = vertices[face.x];
		cvec3  vb = vertices[face.y];
		cvec3  vc = vertices[face.z];

		isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
		isect->primitive = hit_index;
		isect->barys = { barys.x, barys.y };

	}
}


kr_internal kr_global
void soa_obvh_compressed_persistent_intersect(
	const kr_SoA_OBVH_Compressed* __restrict__ bvh,
	const kr_vec3* __restrict__ vertices,
	const kr_uvec4* __restrict__ faces,
	const u32* __restrict__ primitives,
	const kr_ray* __restrict__ rays,
	kr_intersection* isects,
	u32 ray_count,
	u32* warp_counter) {
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool more_work = true;

	vec3 origin;
	vec3 direction;
	vec3 normal;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x76543210;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;

	while (__ballot_sync(WARP_FULL_MASK, more_work == true)) {
		ray_index = atomicAdd(warp_counter, 1);
		more_work = !(ray_index >= ray_count);
		const kr_ray* ray = kr_null;
		kr_intersection* isect = kr_null;

		internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
		leafNodeIdx = 0;
		hit_index = 0xFFFFFFFF;
		stackPtr = 0;

		if (more_work) {
			ray = &rays[ray_index];
			isect = &isects[ray_index];
			isect->primitive = hit_index;
			isect->barys = { 0.5, 0.5 };
			isect->geom_normal = ray->direction;

			origin = ray->origin;
			direction = ray->direction;
			tmax = ray->tmax;
			hitT = tmax;
			internalNodeIdx = 0;
		}

		int active_rays = __ballot_sync(WARP_FULL_MASK, more_work == true);
		while (internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
			bool searchingLeaf = true;

			while (searchingLeaf && internalNodeIdx >= 0 && internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
				const aabb3& lbounds  = bvh->lbounds[internalNodeIdx];
				const aabb3& rbounds  = bvh->rbounds[internalNodeIdx];
				const ivec4& metadata = bvh->metadata[internalNodeIdx];

				kr_vec4 q = {
					(kr_scalar)char(((u32)metadata.z >> 24)) / 127.0f,
					(kr_scalar)char(((u32)metadata.z >> 16)) / 127.0f,
					(kr_scalar)char(((u32)metadata.z >> 8)) / 127.0f,
					((kr_scalar)char(((u32)metadata.z >> 0)) / 127.0f) * KR_PI
				};
				
				cvec2 lisects = kr_ray_unit_aabb_intersect(
					kr_vmul3(kr_vrotate_angle_axis3(direction, { q.x, q.y, q.z }, q.w), lbounds.max),
					kr_vmul3(kr_vrotate_angle_axis3(kr_vsub3(origin, lbounds.min), { q.x, q.y, q.z }, q.w), lbounds.max),
					hitT
				);

				q = {
					(kr_scalar)char(((u32)metadata.w >> 24)) / 127.0f,
					(kr_scalar)char(((u32)metadata.w >> 16)) / 127.0f,
					(kr_scalar)char(((u32)metadata.w >> 8)) / 127.0f,
					((kr_scalar)char(((u32)metadata.w >> 0)) / 127.0f) * KR_PI
				};

				cvec2 risects = kr_ray_unit_aabb_intersect(
					kr_vmul3(kr_vrotate_angle_axis3(direction, { q.x, q.y, q.z }, q.w), rbounds.max),
					kr_vmul3(kr_vrotate_angle_axis3(kr_vsub3(origin, rbounds.min), { q.x, q.y, q.z }, q.w), rbounds.max),
					hitT
				);

				const bool traverseLeft = (lisects.y >= lisects.x) && (lisects.y >= 0.0) && (lisects.x <= hitT);
				const bool traverseRight = (risects.y >= risects.x) && (risects.y >= 0.0) && (risects.x <= hitT);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				if (traverseLeft != traverseRight) {
					if (traverseRight) internalNodeIdx = right_internalNodeIdx;
				}
				else {
					if (!traverseLeft) {
						internalNodeIdx = traversalStack[stackPtr--];
					}
					else {
						if (risects.x < lisects.x) {
							// Swap node IDs
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
					searchingLeaf = false;
					leafNodeIdx = internalNodeIdx;
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			__syncwarp(active_rays);

			while (leafNodeIdx < 0) {
				const ivec4& metadata = bvh->metadata[~leafNodeIdx];
				i32 beg = metadata.x;
				i32 end = metadata.y;
				i32 num = end - beg;
				for (; beg < end; ++beg) {
					u32 primitive_id = (num == 1) ? beg : primitives[beg];

					cuvec4 face = faces[primitive_id];
					cvec3  va = vertices[face.x];
					cvec3  vb = vertices[face.y];
					cvec3  vc = vertices[face.z];

					vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
					if (tisect.z < hitT) {
						hitT = tisect.z;
						hit_index = (u32)primitive_id;
						barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
					}
				}
				leafNodeIdx = internalNodeIdx;
				if (internalNodeIdx < 0)
				{
					internalNodeIdx = traversalStack[stackPtr--];
				}
			}

			active_rays = __ballot_sync(active_rays, internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL);
		}

		if (hit_index == 0xFFFFFFFF)
			continue;

		cuvec4 face = faces[hit_index];
		cvec3  va = vertices[face.x];
		cvec3  vb = vertices[face.y];
		cvec3  vc = vertices[face.z];

		isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
		isect->primitive = hit_index;
		isect->barys = { barys.x, barys.y };
	}
}

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
	u32* warp_counter) {

	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);
	//dim3 blockSize = dim3(block_size);
	//dim3 gridSize = dim3(grid_size);

	soa_obvh_compressed_persistent_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives,
		rays, isects, ray_count, warp_counter);

	/*obvh_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives, transforms,
		rays, isects, ray_count, warp_counter);*/

	return kr_success;
}

kr_internal kr_global
void soa_obvh_persistent_trw_intersect(
	const kr_SoA_OBVH* __restrict__ bvh,
	const kr_vec3* __restrict__ vertices,
	const kr_uvec4* __restrict__ faces,
	const u32* __restrict__ primitives,
	const kr_ray* __restrict__ rays,
	kr_intersection* isects,
	u32 ray_count,
	u32* warp_counter) {
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool more_work = true;

	vec3 origin;
	vec3 direction;
	vec3 normal;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x3FFFFFFF;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;
	int globalIdx = worker_index;

	while (__ballot_sync(WARP_FULL_MASK, more_work == true)) {
		ray_index = atomicAdd(warp_counter, 1);
		more_work = !(ray_index >= ray_count);
		const kr_ray* ray = kr_null;
		kr_intersection* isect = kr_null;

		internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
		leafNodeIdx = 0;
		hit_index = 0xFFFFFFFF;
		stackPtr = 0;

		if (more_work) {
			ray = &rays[ray_index];
			isect = &isects[ray_index];
			isect->barys = { 0.5, 0.5 };
			isect->geom_normal = ray->direction;

			origin = ray->origin;
			direction = ray->direction;
			tmax = ray->tmax;
			hitT = tmax;
			internalNodeIdx = 0;
		}

		int active_rays = __ballot_sync(WARP_FULL_MASK, more_work == true);
		bool searchingOBB = true;

#define isNotEmpty(node) (node & 0xBFFFFFFF) != KR_LBVH_CUDA_STACK_SENTINEL
#define isNotOBB(node) (0x40000000 & node) != 0x40000000
#define markOBB(node) 0x40000000 | node

		while (isNotEmpty(internalNodeIdx)) {
			bool searchingLeaf = true;

			while (searchingOBB && internalNodeIdx >= 0 && isNotEmpty(internalNodeIdx))
			{
				//printf("Searching OBB: Masked Idx %d - Actual idx %d\n", internalNodeIdx, internalNodeIdx & 0xBFFFFFFF);
				const ivec4& metadata = bvh->metadata[internalNodeIdx];
				mat43* left_T = &bvh->left_T[internalNodeIdx];
				mat43* right_T = &bvh->right_T[internalNodeIdx];

				aabb3 left_b;
				aabb3 right_b;

				left_b.min = left_T->cols[0];
				left_b.max = left_T->cols[1];

				right_b.min = right_T->cols[0];
				right_b.max = right_T->cols[1];

				cvec2 lisects = kr_ray_aabb_intersect(direction, origin, left_b, hitT);
				cvec2 risects = kr_ray_aabb_intersect(direction, origin, right_b, hitT);

				const bool traverseLeft = (lisects.y >= lisects.x);
				const bool traverseRight = (risects.y >= risects.x);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				internalNodeIdx = metadata.z == 1 ? markOBB(internalNodeIdx) : internalNodeIdx;
				right_internalNodeIdx = metadata.z == 1 ? markOBB(right_internalNodeIdx) : right_internalNodeIdx;

				//if (globalIdx == 0) printf("%d\n", metadata.z);

				if (traverseLeft != traverseRight) {
					if (traverseRight) internalNodeIdx = right_internalNodeIdx;
				}
				else {
					if (!traverseLeft) { internalNodeIdx = traversalStack[stackPtr--]; }
					else {
						if (risects.x < lisects.x) {
							// Swap node IDs
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						// mark with flag
						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				searchingOBB = isNotOBB(internalNodeIdx);
			}

			//printf("Searching OBB\n");
			__syncwarp(active_rays);

			while (searchingLeaf && internalNodeIdx >= 0 && isNotEmpty(internalNodeIdx))
			{
				//printf("Searching AABB: Masked Idx %d - Actual idx %d\n", internalNodeIdx, internalNodeIdx & 0xBFFFFFFF);

				internalNodeIdx &= 0xBFFFFFFF;//clear flag
				const ivec4& metadata = bvh->metadata[internalNodeIdx];
				const mat43* left_T = &bvh->left_T[internalNodeIdx];
				const mat43* right_T = &bvh->right_T[internalNodeIdx];

				cvec2 lisects = kr_ray_unit_aabb_intersect(kr_n43transform3p(left_T, &direction), kr_v43transform3p(left_T, &origin), hitT);
				cvec2 risects = kr_ray_unit_aabb_intersect(kr_n43transform3p(right_T, &direction), kr_v43transform3p(right_T, &origin), hitT);

				const bool traverseLeft = (lisects.y >= lisects.x);
				const bool traverseRight = (risects.y >= risects.x);

				internalNodeIdx = metadata.x;
				int right_internalNodeIdx = metadata.y;

				internalNodeIdx = internalNodeIdx > 0 ? markOBB(internalNodeIdx) : internalNodeIdx;
				right_internalNodeIdx = right_internalNodeIdx > 0 ? markOBB(right_internalNodeIdx) : right_internalNodeIdx;

				if (traverseLeft != traverseRight) { if (traverseRight) internalNodeIdx = right_internalNodeIdx; }
				else {
					if (!traverseLeft) {  internalNodeIdx = traversalStack[stackPtr--]; }
					else {
						if (risects.x < lisects.x) {
							int tmp = internalNodeIdx;
							internalNodeIdx = right_internalNodeIdx;
							right_internalNodeIdx = tmp;
						}

						traversalStack[++stackPtr] = right_internalNodeIdx;
					}
				}

				if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
					searchingLeaf = false;
					leafNodeIdx = internalNodeIdx;
					internalNodeIdx = traversalStack[stackPtr--];
				}

				if (internalNodeIdx > 0 && isNotOBB(internalNodeIdx)) { searchingOBB = true; }
			}

			__syncwarp(active_rays);

			/**/
			if (internalNodeIdx < 0)
			{
				while(internalNodeIdx < 0) internalNodeIdx = traversalStack[stackPtr--];
				if (internalNodeIdx > 0) { searchingOBB = isNotOBB(internalNodeIdx) ? true : false; }
			}
			active_rays = __ballot_sync(active_rays, isNotEmpty(internalNodeIdx));
			continue;

			while (leafNodeIdx < 0) {
				printf("Masked Idx %d\n", leafNodeIdx);
				const ivec4& metadata = bvh->metadata[~leafNodeIdx];
				i32 beg = metadata.x;
				i32 end = metadata.y;
				i32 num = end - beg;

				for (; beg < end; ++beg) {
					u32 primitive_id = (num == 1) ? beg : primitives[beg];

					cuvec4 face = faces[primitive_id];
					cvec3  va = vertices[face.x];
					cvec3  vb = vertices[face.y];
					cvec3  vc = vertices[face.z];

					vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
					if (tisect.z < hitT) {
						hitT = tisect.z;
						hit_index = (u32)primitive_id;
						barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
					}
				}

				leafNodeIdx = internalNodeIdx;

				if (internalNodeIdx < 0)
				{
					internalNodeIdx = traversalStack[stackPtr--];
					if (internalNodeIdx > 0) { searchingOBB = isNotOBB(internalNodeIdx) ? true : false; }
				}
			}

			active_rays = __ballot_sync(active_rays, isNotEmpty(internalNodeIdx));
		}

		if (hit_index == 0xFFFFFFFF)
			continue;

		cuvec4 face = faces[hit_index];
		cvec3  va = vertices[face.x];
		cvec3  vb = vertices[face.y];
		cvec3  vc = vertices[face.z];

		isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
		isect->primitive = hit_index;
		isect->barys = { barys.x, barys.y };
	}
}

kr_error kr_cuda_obvh_intersect(
	const kr_bvh_node* bvh,
	const kr_vec3* vertices,
	const kr_uvec4* faces,
	const u32* primitives,
	const kr_mat4* transforms,
	const kr_ray* rays,
	kr_intersection* isects,
	u32 ray_count) {
	
	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);

	obvh_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives, transforms,
		rays, isects, ray_count);

	return kr_success;
}

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
	u32* warp_counter) {

	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);
	//dim3 blockSize = dim3(block_size);
	//dim3 gridSize = dim3(grid_size);

	obvh_persistent_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives, transforms,
		rays, isects, ray_count, warp_counter);

	/*obvh_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives, transforms,
		rays, isects, ray_count, warp_counter);*/

	return kr_success;
}

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
	u32* warp_counter) {

	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);
	
	soa_obvh_persistent_intersect << < gridSize, blockSize >> > (
		bvh, vertices, faces, primitives,
		rays, isects, ray_count, warp_counter); 

	return kr_success;
}

kr_error kr_cuda_obvh_intersect_ray(
	const kr_bvh_node* bvh,
	const kr_vec3* vertices,
	const kr_uvec4* faces,
	const u32* primitives,
	const kr_mat4* transforms,
	const kr_ray* ray,
	kr_intersection* isect) {
		
	return kr_success;
}

kr_error kr_cuda_bvh_intersect_bounds_ray(
	const kr_bvh_node* bvh,
	const kr_ray* ray,
	kr_intersection* isect, i32 max_level){
		
	return kr_success;
}


kr_internal kr_global
void bvh_intersect_bounds(
	const kr_bvh_node* bvh,
	const kr_mat4* transforms,
	const kr_ray* rays,
	kr_intersection* isects,
	u32 ray_count, i32 max_level) {
	const u32 ray_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_index >= ray_count)
		return;

	const kr_bvh_node* nodes = bvh;
	const kr_ray* ray = rays + ray_index;

	kr_intersection* isect = isects + ray_index;
	kr_scalar min_distance = ray->tmax;
	isect->primitive = kr_invalid_index;

	vec3 origin = ray->origin;
	vec3 direction = ray->direction;
	vec3 idirection = kr_vinverse3(ray->direction);
	vec3 normal = { 0, 0, 0 };

	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;

	i32 toVisitOffset = 0, currentNodeIndex = 0, currentNodeLevel = 0;
	i32 nodesToVisit[KR_LBVH_CUDA_STACK_SIZE] = { 0 };
	i32 nodeLevels[KR_LBVH_CUDA_STACK_SIZE] = { 0 };
	i32 maxLevel = (max_level < 0) ? 10000 : max_level;
	
	while (kr_true) {
		const kr_bvh_node* node = &nodes[currentNodeIndex];
		vec2 isects = { 0 };
	
		isects = kr_ray_aabb_intersect_n(direction, origin, node->bounds, min_distance, &normal);

		if ((isects.x < 0 && isects.y < 0)
			|| isects.x >= min_distance) {
			if (toVisitOffset == 0) break;
			--toVisitOffset;
			currentNodeIndex = nodesToVisit[toVisitOffset];
			currentNodeLevel = nodeLevels[toVisitOffset];
			continue;
		}

		if (maxLevel == currentNodeLevel) {
			if (isects.x < min_distance) {
				min_distance = isects.x;
				isect->primitive = 0;
				isect->instance = 0;
				isect->geom_normal = normal;
				isect->barys = KR_INITIALIZER_CAST(vec2) { 0, 0 };
				//first_hit_object = instance;
			}

			if (toVisitOffset == 0) break;
			--toVisitOffset;
			currentNodeIndex = nodesToVisit[toVisitOffset];
			currentNodeLevel = nodeLevels[toVisitOffset];
			continue;
		}

		if (node->nPrimitives > 0) {
			if (isects.x < min_distance) {
				min_distance = isects.x;
				isect->primitive = 0;
				isect->instance = 0;
				isect->geom_normal = normal;
				isect->barys = KR_INITIALIZER_CAST(vec2) { 0, 0 };
				//first_hit_object = instance;
			}
			if (toVisitOffset == 0) break;
			--toVisitOffset;
			currentNodeIndex = nodesToVisit[toVisitOffset];
			currentNodeLevel = nodeLevels[toVisitOffset];
		}
		else {
			const kr_bvh_node* l = &nodes[node->left];
			const kr_bvh_node* r = &nodes[node->right];
			vec2 lisects = { 0 };
			vec2 risects = { 0 };

			if (l->axis == 1 && kr_null != transforms) {
				const kr_mat4* ltransform = &transforms[node->left];
				cvec3 lorigin_lcs = kr_vtransform3p(ltransform, &origin);
				cvec3 ldirection_lcs = kr_ntransform3p(ltransform, &direction);
				lisects = kr_ray_unit_aabb_intersect_n(ldirection_lcs, lorigin_lcs, min_distance, &normal);
			}
			else {
			
				lisects = kr_ray_aabb_intersect_n(direction, origin, kr_aabb_shrink3(l->bounds, 0.0), min_distance, &normal);
			}

			if (r->axis == 1 && kr_null != transforms) {
				const kr_mat4* rtransform = &transforms[node->right];
				cvec3 rorigin_lcs = kr_vtransform3p(rtransform, &origin);
				cvec3 rdirection_lcs = kr_ntransform3p(rtransform, &direction);
				risects = kr_ray_unit_aabb_intersect_n(rdirection_lcs, rorigin_lcs, min_distance, &normal);
			}
			else {
				risects = kr_ray_aabb_intersect_n(direction, origin, kr_aabb_shrink3(r->bounds, 0.0), min_distance, &normal);
			}


			b32 traverseChild0 = (lisects.y >= lisects.x);
			b32 traverseChild1 = (risects.y >= risects.x);
			if (traverseChild0 != traverseChild1) {
				if (traverseChild0) {
					nodeLevels[toVisitOffset] = currentNodeLevel + 1;
					nodesToVisit[toVisitOffset++] = node->right;
					currentNodeIndex = node->left;
					currentNodeLevel++;
				}
				else {
					nodeLevels[toVisitOffset] = currentNodeLevel + 1;
					nodesToVisit[toVisitOffset++] = node->left;
					currentNodeIndex = node->right;
					currentNodeLevel++;
				}
			}
			else {
				if (!traverseChild0) {
					if (toVisitOffset == 0) break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
					continue;
				}
				else {
					if (lisects.x < risects.x) {
						nodeLevels[toVisitOffset] = currentNodeLevel + 1;
						nodesToVisit[toVisitOffset++] = node->right;
						currentNodeIndex = node->left;
						currentNodeLevel++;
					}
					else {
						nodeLevels[toVisitOffset] = currentNodeLevel + 1;
						nodesToVisit[toVisitOffset++] = node->left;
						currentNodeIndex = node->right;
						currentNodeLevel++;
					}
				}
			}
		}
	}
}

kr_error kr_cuda_bvh_intersect_bounds(
	const kr_bvh_node* bvh,
	const kr_ray* rays,
	kr_intersection* isects,
	u32 ray_count, i32 max_level){
	
	kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
		dim3 blockSize = dim3(128);
		int bx = (ray_count + blockSize.x - 1) / blockSize.x;
		dim3 gridSize = dim3(bx);
		bvh_intersect_bounds <<< gridSize, blockSize >> > (
			bvh, kr_null, rays, isects,
			ray_count, max_level);
		});
	kr_log("bvh bounds intersection %fms\n", elapsed_ms);

	return kr_success;
}

kr_error kr_cuda_obvh_intersect_bounds(
	const kr_bvh_node* bvh,
	const kr_mat4* transforms,
	const kr_ray* rays,
	kr_intersection* isects,
	u32 ray_count, i32 max_level) {

	dim3 blockSize = dim3(128);
	int bx = (ray_count + blockSize.x - 1) / blockSize.x;
	dim3 gridSize = dim3(bx);
	bvh_intersect_bounds << < gridSize, blockSize >> > (
		bvh, transforms, rays, isects,
		ray_count, max_level);

	return kr_success;
}

kr_inline_device
void bvh_cuda_persistent_intersect_ray(
	kr_handle tlas,
	const kr_ray* ray,
	kr_intersection* isect)
{
#if 0
	const kr_cuda_blas* blas = (kr_cuda_blas*)tlas,
	const kr_vec3* vertices = blas->vertices;
	const kr_uvec4* faces = blas->faces;
	const u32* primitives = blas->primitives;
	const kr_bvh_node_packed* nodes = blas->bvh;
	const u32 worker_index = blockIdx.x * blockDim.x + threadIdx.x;

	vec3 origin;
	vec3 direction;
	vec3 barys;

	kr_scalar tmin;
	kr_scalar tmax;
	kr_scalar hitT;

	constexpr auto KR_LBVH_CUDA_STACK_SIZE = 64;
	constexpr i32 KR_LBVH_CUDA_STACK_SENTINEL = 0x76543210;

	i32 stackPtr;
	i32 internalNodeIdx;
	i32 leafNodeIdx;
	i32 traversalStack[KR_LBVH_CUDA_STACK_SIZE];
	u32 ray_index, hit_index;

	traversalStack[0] = KR_LBVH_CUDA_STACK_SENTINEL;

	internalNodeIdx = KR_LBVH_CUDA_STACK_SENTINEL;
	leafNodeIdx = 0;
	hit_index = 0xFFFFFFFF;
	stackPtr = 0;

	origin = ray->origin;
	direction = ray->direction;
	tmax = ray->tmax;
	hitT = tmax;
	internalNodeIdx = 0;

	while (internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
		bool searchingLeaf = true;

		while (searchingLeaf && internalNodeIdx >= 0 && internalNodeIdx != KR_LBVH_CUDA_STACK_SENTINEL) {
			const kr_aabb3& lbounds = nodes[internalNodeIdx].lbounds;
			const kr_aabb3& rbounds = nodes[internalNodeIdx].rbounds;
			const ivec4& metadata = nodes[internalNodeIdx].metadata;

			cvec2 risects = kr_ray_aabb_intersect(direction, origin, rbounds, hitT);
			cvec2 lisects = kr_ray_aabb_intersect(direction, origin, lbounds, hitT);

			bool traverseLeft = (lisects.y >= lisects.x);
			bool traverseRight = (risects.y >= risects.x);

			internalNodeIdx = metadata.x;
			int right_internalNodeIdx = metadata.y;

			if (traverseLeft != traverseRight) {
				if (traverseRight) internalNodeIdx = right_internalNodeIdx;
			}
			else {
				if (!traverseLeft) {
					internalNodeIdx = traversalStack[stackPtr--];
				}
				else {
					if (risects.x < lisects.x) {
						// Swap node IDs
						int tmp = internalNodeIdx;
						internalNodeIdx = right_internalNodeIdx;
						right_internalNodeIdx = tmp;
					}

					traversalStack[++stackPtr] = right_internalNodeIdx;
				}
			}

			if (internalNodeIdx < 0 && leafNodeIdx >= 0) {
				searchingLeaf = false;
				leafNodeIdx = internalNodeIdx;
				internalNodeIdx = traversalStack[stackPtr--];
			}
		}

		while (leafNodeIdx < 0) {
			const ivec4& metadata = nodes[~leafNodeIdx].metadata;
			i32 beg = metadata.x;
			i32 end = metadata.y;
			i32 num = end - beg;
			for (; beg < end; ++beg) {
				u32 primitive_id = (num == 1) ? beg : primitives[beg];

				cuvec4 face = faces[primitive_id];
				cvec3  va = vertices[face.x];
				cvec3  vb = vertices[face.y];
				cvec3  vc = vertices[face.z];

				vec3 tisect = kr_ray_triangle_intersect(direction, origin, va, vb, vc, hitT);
				if (tisect.z < hitT) {
					hitT = tisect.z;
					hit_index = (u32)primitive_id;
					barys = KR_INITIALIZER_CAST(vec2) { tisect.x, tisect.y };
				}
			}
			leafNodeIdx = internalNodeIdx;
			if (internalNodeIdx < 0)
			{
				internalNodeIdx = traversalStack[stackPtr--];
			}
		}
	}

	if (hit_index == 0xFFFFFFFF)
		return;

	cuvec4 face = faces[hit_index];
	cvec3  va = vertices[face.x];
	cvec3  vb = vertices[face.y];
	cvec3  vc = vertices[face.z];

	isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));
	isect->primitive = hit_index;
	isect->barys = { barys.x, barys.y };
#endif
}
