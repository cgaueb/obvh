#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "common/korangar.h"
#include "common/ads.h"

#include "lbvh.h"

#include "common/logger.h" 
#include "common/queue.h" 
#include "common/geometry.h" 
#include "common/util.h" 
#include "common/cuda/util.h"
#include "common/cuda/atomics.h"

#include "common/algorithm/dito/dito.cuh"

#include "ads/common/cuda/intersectors.cuh"
#include "ads/common/cuda/transforms.cuh"
#include "ads/common/transforms.cpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <map>
#include <vector>
#include <list>
#include <chrono>

struct kr_obb_pair {
    kr_obb3 lb;
    kr_obb3 ub;
};

kr_error
lbvh_cuda_query_intersection(kr_ads_lbvh_cuda* ads, kr_intersection_query* query) {
    const kr_ads_blas_cuda* blas = &ads->blas;
    const kr_object_cu* object = blas->h_object_cu;

    thrust::fill(thrust::device, blas->ray_counter, blas->ray_counter + 1, 0);

    if (ads->intersect_bounds) {
        const kr_bvh_node* bvh = blas->bvh;
        if (!ads->use_obbs) {
            return kr_cuda_bvh_intersect_bounds(
                bvh,
                query->cuda.rays,
                query->cuda.isects,
                query->cuda.ray_count, ads->intersect_bounds_level);
        }
        else {
            return kr_cuda_obvh_intersect_bounds(
                bvh,
                blas->obb_transforms,
                query->cuda.rays,
                query->cuda.isects,
                query->cuda.ray_count, ads->intersect_bounds_level);
        }
    }

    if (ads->use_persistent_kernel) {
        const kr_bvh_node_packed* bvh = blas->bvh_packed;
        if (!ads->use_obbs) {
            return kr_cuda_bvh_persistent_intersect(bvh,
                object->as_mesh.vertices,
                object->as_mesh.faces,
                blas->primitives,
                query->cuda.rays,
                query->cuda.isects,
                query->cuda.ray_count,
                ads->cu_blockSize,
                ads->cu_gridSize,
                blas->ray_counter);
        }
        else {
            /*return kr_cuda_obvh_persistent_intersect(bvh,
                object->as_mesh.vertices,
                object->as_mesh.faces,
                blas->primitives,
                blas->transformations,
                query->cuda.rays,
                query->cuda.isects,
                query->cuda.ray_count,
                ads->cu_blockSize,
                ads->cu_gridSize,
                blas->ray_counter);*/


            return kr_cuda_soa_obvh_intersect(blas->soa_obvh_packed,
                object->as_mesh.vertices,
                object->as_mesh.faces,
                blas->primitives,
                query->cuda.rays,
                query->cuda.isects,
                query->cuda.ray_count,
                ads->cu_blockSize,
                ads->cu_gridSize,
                blas->ray_counter);
        }
    }
    else {
        const kr_bvh_node* bvh = (blas->collapsed_bvh) ? blas->collapsed_bvh : blas->bvh;

        if (!ads->use_obbs) {
            return kr_success;
        }
        else {
            return kr_cuda_obvh_intersect(bvh,
                object->as_mesh.vertices,
                object->as_mesh.faces,
                blas->primitives,
                blas->obb_transforms,
                query->cuda.rays,
                query->cuda.isects,
                query->cuda.ray_count);
        }
    }

    return kr_success;
}

kr_ads_lbvh_cuda*
lbvh_cuda_create() {
    return (kr_ads_lbvh_cuda*)kr_allocate(sizeof(kr_ads_lbvh_cuda));
}

__global__ void
calculate_collapsed_node_costs_kernel(
    const kr_object_cu* object,
    const kr_bvh_node* nodes,
    const u32* parents,
    kr_scalar* costs,
    const b32* collapse_table,
    b32* visit_table
) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    const u32 node_count = primitive_count + internal_count;
    if (node_index >= node_count)
        return;

    i32 i = node_index;

    const kr_bvh_node* node = &nodes[i];
    kr_scalar ct = 1.0;
    kr_scalar ci = 1.2;

    if (i < internal_count) {
        if (node->nPrimitives == 0) return;
    }
    else {
        if (kr_true == collapse_table[i]) return;
    }

    const kr_bvh_node* parent;
    kr_u32 parent_index = parents[i];

    while (parent_index != 0xFFFFFFFF) {
        parent = &nodes[parent_index];

        b32 visited = atomicExch(&visit_table[parent_index], kr_true);
        __threadfence();
        if (kr_false == visited) {
            break;
        }

        kr_scalar cost_left = costs[parent->left];
        kr_scalar cost_right = costs[parent->right];

        aabb3 box_node = parent->bounds;
        costs[parent_index] = ci * kr_aabb_surface_area3(box_node) + cost_right + cost_left;

        parent_index = parents[parent_index];
    }
}

__global__ void
calculate_collapsed_leaf_costs_kernel(
    const kr_object_cu* object,
    const kr_bvh_node* nodes,
    kr_scalar* costs,
    const b32* collapse_table
) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    const u32 node_count = primitive_count + internal_count;
    if (node_index >= node_count)
        return;

    i32 i = node_index;

    const kr_bvh_node* node = &nodes[i];
    kr_scalar ct = 1.0f;
    kr_scalar ci = 1.2f;

    if (i < internal_count) {
        if (node->nPrimitives == 0) return;

        costs[i] = ct * node->nPrimitives * kr_aabb_surface_area3(node->bounds);
    }
    else {
        if (kr_true == collapse_table[i]) return;

        costs[i] = ct * kr_aabb_surface_area3(node->bounds);
    }

}

kr_internal kr_error
lbvh_cuda_collapse(kr_ads_lbvh_cuda* ads, kr_ads_blas_cuda* blas, kr_ads_build_metrics * metrics) {
    cudaError_t cu_error;
    kr_object_cu* d_object = blas->d_object_cu;
    kr_object_cu* h_object = blas->h_object_cu;
    
    return kr_cuda_bvh_collapse(
        blas->bvh,
        blas->parents,
            blas->collapsed_bvh,
        blas->primitive_counts,
        blas->primitives,
            blas->costs,
        ads->cost_internal,
        ads->cost_traversal,
        blas->leaf_count,
        blas->internal_count,
        &metrics->collapse_metrics);
   
    return kr_success;
}

kr_internal kr_inline_device kr_morton
kr_vmorton3(vec3 a) {
#ifdef KR_MORTON_64
    return kr_v64morton3(a);
#else
    return kr_v32morton3(a);
#endif
}

__global__ void 
mortons_kernel(const kr_object_cu* object, kr_morton* mortons, kr_bvh_node* leaf_nodes, const aabb3* aabbs) {
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index >= object->as_mesh.face_count)
        return;

    // Get centroid AABB
    aabb3 cell = *(aabbs + 1);

    kr_bvh_node* leaf_node = leaf_nodes + primitive_index;

    cuvec4 face = object->as_mesh.faces[primitive_index];
    cvec3  va   = object->as_mesh.vertices[face.x];
    cvec3  vb   = object->as_mesh.vertices[face.y];
    cvec3  vc   = object->as_mesh.vertices[face.z];
    aabb3 bbox = kr_aabb_create3(va, va);

    bbox = kr_aabb_expand3(bbox, vb);
    bbox = kr_aabb_expand3(bbox, vc);

    leaf_node->bounds = bbox;

    bbox = kr_aabb_offset3(bbox, kr_vnegate3(cell.min));
    vec3 center = kr_aabb_center3(bbox);
    center = kr_vdiv3s(center, kr_vsub3(cell.max, cell.min));
    kr_morton morton = kr_vmorton3(center);
    mortons[primitive_index] = morton;

    leaf_node->nPrimitives = 1;
    leaf_node->primitivesOffset = primitive_index;
}

kr_internal __inline__ __device__ kr_morton
kr_morton_clz(kr_morton id) {
    return __clz(id);
}

kr_internal __inline__ __device__ kr_i32
delta(const kr_morton* codes, kr_i32 i, kr_i32 j, kr_u32 count) {
    if (j > (kr_i32)count - 1 || j < 0)
        return -1;

    kr_morton id_i = codes[i];
    kr_morton id_j = codes[j];

#ifdef KR_MORTON_64
#define KR_CLASH_RESOLVE_BASE 64
#else
#define KR_CLASH_RESOLVE_BASE 32
#endif

    /* Clever way to properly work even when there are duplicate node IDs */
    if (id_i == id_j)
        return (kr_i32)(KR_CLASH_RESOLVE_BASE + kr_morton_clz((kr_morton)i ^ (kr_morton)j));

    return (kr_i32)kr_morton_clz(id_i ^ id_j);
}

__device__ kr_i32
least_integer_divide(kr_i32 num) {
    if (num == 1) return 0;
    else if ((num % 2) == 1) return num / 2 + 1;
    else return num / 2;
}

__global__ void
radix_tree_build_kernel(const kr_object_cu* object, kr_bvh_node* nodes, const kr_morton* mortons, u32* parents) {
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (primitive_index >= internal_count)
        return;

    const kr_morton* codes = mortons;
    kr_bvh_node* internal_nodes = nodes;
    kr_bvh_node* leaf_nodes = internal_nodes + internal_count;

    i32 i = primitive_index;

    kr_morton id = mortons[i];

    kr_i32 delta_a = delta(codes, i, i + 1, primitive_count);
    kr_i32 delta_b = delta(codes, i, i - 1, primitive_count);

    kr_i32 d = kr_signi(delta_a - delta_b);

    kr_i32 delta_min = delta(codes, i, i - d, primitive_count);

    kr_i32 l_max = 2;

    while (delta(codes, i, i + l_max * d, primitive_count) > delta_min) {
        l_max *= 2;
    }

    kr_i32 l = 0;

    kr_i32 t = l_max / 2;
    while (t >= 1) {
        kr_i32 delt = delta(codes, i, i + (l + t) * d, primitive_count);
        if (delt > delta_min) {
            l = l + t;
        }

        t /= 2;
    }

    kr_i32 j = i + l * d;

    kr_i32 delta_node = delta(codes, i, j, primitive_count);

    kr_i32 s = 0;
    t = least_integer_divide(l);
    while (t >= 1) {
        kr_i32 delt = delta(codes, i, i + (s + t) * d, primitive_count);
        if (delt > delta_node) {
            s = s + t;
        }
        t = least_integer_divide(t);
    }

    kr_i32 split_position = i + s * d + kr_mini(d, 0);

    b32 left_is_leaf = (kr_mini(i, j) == (split_position + 0)) ? kr_true : kr_false;
    b32 right_is_leaf = (kr_maxi(i, j) == (split_position + 1)) ? kr_true : kr_false;

    u32 left_index = split_position + 0 + (left_is_leaf ? internal_count : 0);
    u32 right_index = split_position + 1 + (right_is_leaf ? internal_count : 0);

    parents[left_index] = i;
    parents[right_index] = i;

    u32 split = (kr_morton_clz(codes[left_index - (left_is_leaf ? internal_count : 0)] ^ codes[right_index - (right_is_leaf ? internal_count : 0)]) - 2) % 3;

    //internal_nodes[i].axis = split;
    internal_nodes[i].axis = 0;
    internal_nodes[i].left = left_index;
    internal_nodes[i].right = right_index;
    internal_nodes[i].nPrimitives = 0;
    internal_nodes[i].bounds = kr_aabb_empty3();


    if (0 == i) {
        parents[0] = (u32)-1;
    }
}


__global__ void
calculate_aabbs_kernel(const kr_object_cu* object, kr_bvh_node* nodes, u32* parents, u32* primitive_counts, kr_scalar* costs, b32* visit_table) {
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (primitive_index >= primitive_count)
        return;

    i32 i = primitive_index;

    kr_bvh_node* internal_nodes = nodes;
    kr_bvh_node* leaf_nodes = internal_nodes + internal_count;

    u32* internal_parents = parents;
    u32* leaf_parents = internal_parents + internal_count;

    u32* internal_counts = primitive_counts;
    u32* leaf_counts = internal_counts + internal_count;

    kr_scalar* internal_costs = costs;
    kr_scalar* leaf_costs = internal_costs + internal_count;

    kr_scalar ct = 1.0;
    kr_scalar ci = 1.2;

    kr_bvh_node* leaf = &leaf_nodes[i];
    kr_bvh_node* parent;

    leaf_costs[i] = ct * kr_aabb_surface_area3(leaf->bounds);
    leaf_counts[i] = 1;

    kr_u32 parent_index = leaf_parents[i];

    while (parent_index != 0xFFFFFFFF) {
        parent = &internal_nodes[parent_index];

        b32 visited = atomicExch(&visit_table[parent_index], kr_true);
        __threadfence();
        if (kr_false == visited) {
            break;
        }

        aabb3 box_left = nodes[parent->left].bounds;
        aabb3 box_right = nodes[parent->right].bounds;
        aabb3 box_node = kr_aabb_expand(box_left, box_right);

        kr_scalar cost_left = costs[parent->left];
        kr_scalar cost_right = costs[parent->right];

        u32 count_left = primitive_counts[parent->left];
        u32 count_right = primitive_counts[parent->right];

        primitive_counts[parent_index] = count_left + count_right;
        costs[parent_index] = ci * kr_aabb_surface_area3(box_node) + cost_right + cost_left;
        parent->bounds = box_node;

        parent_index = parents[parent_index];
    }
}

kr_internal std::vector<kr_bvh_node*>
lbvh_node_leaves(kr_bvh_node* nodes, kr_bvh_node* root) {
    std::vector<kr_bvh_node*> leaves;
    std::list<kr_bvh_node*> queue;
    queue.push_back(root);

    while (!queue.empty()) {
        kr_bvh_node* node = queue.front();
        queue.pop_front();

        if (node->nPrimitives > 0) {
            leaves.push_back(node);
            continue;
        }

        queue.push_back(&nodes[node->left]);
        queue.push_back(&nodes[node->right]);
    }

    return leaves;
}

kr_inline_host_device
kr_scalar lbvh_dito14_obb_quality(cvec3 len) {
    return len.x * len.y + len.x * len.z + len.y * len.z; //half box area
}

template <int K = 7>
kr_internal kr_inline_device kr_error
lbvh_dito14_fixed_projections_resolve(
    kr_scalar* o_min_proj, kr_scalar* o_max_proj, kr_minmax_pair* o_minmax_pairs,
    const kr_scalar* l_min_proj, const kr_scalar* l_max_proj, const kr_minmax_pair* l_minmax_pairs,
    const kr_scalar* r_min_proj, const kr_scalar* r_max_proj, const kr_minmax_pair* r_minmax_pairs) {

    for (int i = 0; i < K; i++) {
        if (l_min_proj[i] < r_min_proj[i]) {
            o_min_proj[i] = l_min_proj[i];
            o_minmax_pairs[i].min = l_minmax_pairs[i].min;
        }
        else {
            o_min_proj[i] = r_min_proj[i];
            o_minmax_pairs[i].min = r_minmax_pairs[i].min;
        }
        if (l_max_proj[i] > r_max_proj[i]) {
            o_max_proj[i] = l_max_proj[i];
            o_minmax_pairs[i].max = l_minmax_pairs[i].max;
        }
        else {
            o_max_proj[i] = r_max_proj[i];
            o_minmax_pairs[i].max = r_minmax_pairs[i].max;
        }
    }

    return kr_success;
}

template <int K = 7, b32 PreInitialized = false>
kr_internal kr_inline_device kr_error
lbvh_dito14_fixed_projections(
    kr_scalar* min_proj, kr_scalar* max_proj, kr_minmax_pair* minmax_pairs,
    cvec3* points, kr_size point_count) {
    const auto& point = points[0];
    auto& p = minmax_pairs;

    // Slab 0: dir {1, 0, 0}
    kr_scalar proj = point.x;
    min_proj[0] = max_proj[0] = proj;
    p[0].min = p[0].max = point;
    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    min_proj[1] = max_proj[1] = proj;
    p[1].min = p[1].max = point;
    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    min_proj[2] = max_proj[2] = proj;
    p[2].min = p[2].max = point;
    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    min_proj[3] = max_proj[3] = proj;
    p[3].min = p[3].max = point;
    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    min_proj[4] = max_proj[4] = proj;
    p[4].min = p[4].max = point;
    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    min_proj[5] = max_proj[5] = proj;
    p[5].min = p[5].max = point;
    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    min_proj[6] = max_proj[6] = proj;
    p[6].min = p[6].max = point;

    for (size_t i = 1; i < point_count; i++) {
        const auto& point = points[i];
        // Slab 0: dir {1, 0, 0}
        proj = point.x;
        if (proj < min_proj[0]) { min_proj[0] = proj; p[0].min = point; }
        if (proj > max_proj[0]) { max_proj[0] = proj; p[0].max = point; }
        // Slab 1: dir {0, 1, 0}
        proj = point.y;
        if (proj < min_proj[1]) { min_proj[1] = proj; p[1].min = point; }
        if (proj > max_proj[1]) { max_proj[1] = proj; p[1].max = point; }
        // Slab 2: dir {0, 0, 1}
        proj = point.z;
        if (proj < min_proj[2]) { min_proj[2] = proj; p[2].min = point; }
        if (proj > max_proj[2]) { max_proj[2] = proj; p[2].max = point; }
        // Slab 3: dir {1, 1, 1}
        proj = point.x + point.y + point.z;
        if (proj < min_proj[3]) { min_proj[3] = proj; p[3].min = point; }
        if (proj > max_proj[3]) { max_proj[3] = proj; p[3].max = point; }
        // Slab 4: dir {1, 1, -1}
        proj = point.x + point.y - point.z;
        if (proj < min_proj[4]) { min_proj[4] = proj; p[4].min = point; }
        if (proj > max_proj[4]) { max_proj[4] = proj; p[4].max = point; }
        // Slab 5: dir {1, -1, 1}
        proj = point.x - point.y + point.z;
        if (proj < min_proj[5]) { min_proj[5] = proj; p[5].min = point; }
        if (proj > max_proj[5]) { max_proj[5] = proj; p[5].max = point; }
        // Slab 6: dir {1, -1, -1}
        proj = point.x - point.y - point.z;
        if (proj < min_proj[6]) { min_proj[6] = proj; p[6].min = point; }
        if (proj > max_proj[6]) { max_proj[6] = proj; p[6].max = point; }
    }
    return kr_success;
}


template <int K = 7>
kr_internal kr_inline_device kr_scalar
lbvh_dito14_furthest_point_from_edge(
    cvec3* points, kr_size point_count,
    vec3 p0, vec3 e0, vec3* p)
{
    kr_scalar dist2, max_dist2;
    int at = 0;

    max_dist2 = kr_vdistance_to_inf_edge3(points[0], p0, e0);
    //printf("GPU Infinite Edge %d %f\n", point_count, max_dist2);
    for (int i = 1; i < point_count; i++)
    {
        //kr_vec3 u0 = kr_vsub3(points[i], p0);
        //kr_scalar t = kr_vdot3(e0, u0);
        //kr_scalar sqLen_v = kr_vlength3sqr(e0);
        dist2 = kr_vdistance_to_inf_edge3(points[i], p0, e0);
        //printf("GPU Infinite Edge Vertex {%f %f %f} %f {%f %f}\n", points[i].x, points[i].y, points[i].z, dist2, sqLen_v, t);
        if (dist2 > max_dist2)
        {
            max_dist2 = dist2;
            at = i;
        }
    }

    *p = points[at];

    //printf("GPU Infinite Edge Vertex {%f %f %f} %f\n", p->x, p->y, p->z, max_dist2);

    return max_dist2;
}

template <int K = 7>
kr_internal kr_inline_device kr_error
lbvh_dito14_furthest_point_pair(
    const kr_minmax_pair* minmax_pairs,
    vec3* p0, vec3* p1) {
    const kr_minmax_pair* p = minmax_pairs;
    int at = 0;
    kr_scalar dist2, max_dist2;
    max_dist2 = kr_vdistance3sqr(minmax_pairs[0].min, minmax_pairs[0].max);
    //printf("GPU SqDistance[%d] %f Min: {%f %f %f} Max: {%f %f %f}\n", 0, maxSqDist, minVert[0].x, minVert[0].y, minVert[0].z, maxVert[0].x, maxVert[0].y, maxVert[0].z);
    for (int k = 1; k < K; k++)
    {
        dist2 = kr_vdistance3sqr(minmax_pairs[k].min, minmax_pairs[k].max);
        //printf("GPU SqDistance[%d] %f Min: {%f %f %f} Max: {%f %f %f}\n", k, sqDist, minVert[k].x, minVert[k].y, minVert[k].z, maxVert[k].x, maxVert[k].y, maxVert[k].z);
        if (dist2 > max_dist2) { max_dist2 = dist2; at = k; }
    }

    *p0 = minmax_pairs[at].min;
    *p1 = minmax_pairs[at].max;

    return kr_success;
}

template <int K = 7>
kr_internal kr_inline_device int
lbvh_dito14_base_triangle_construct(
    const kr_minmax_pair* minmax_pairs,
    cvec3* points, kr_size point_count,
    vec3* n, vec3* p0, vec3* p1, vec3* p2,
    vec3* e0, vec3* e1, vec3* e2) {
    kr_scalar dist2;
    kr_scalar eps = 0.000001f;

    // Find the furthest point pair among the selected min and max point pairs
    lbvh_dito14_furthest_point_pair(minmax_pairs, p0, p1);

    // Degenerate case 1:s
    // If the found furthest points are located very close, return OBB aligned with the initial AABB 
    if (kr_vdistance3sqr(*p0, *p1) < eps) { return 1; }

    // Compute edge vector of the line segment p0, p1 		
    *e0 = kr_vnormalize3(kr_vsub3(*p0, *p1));

    // Find a third point furthest away from line given by p0, e0 to define the large base triangle
    dist2 = lbvh_dito14_furthest_point_from_edge(points, point_count, *p0, *e0, p2);

    // Degenerate case 2:
    // If the third point is located very close to the line, return an OBB aligned with the line 
    if (dist2 < eps) { return 2; }

    // Compute the two remaining edge vectors and the normal vector of the base triangle				
    *e1 = kr_vnormalize3(kr_vsub3(*p1, *p2));
    *e2 = kr_vnormalize3(kr_vsub3(*p2, *p0));
    *n = kr_vnormalize3(kr_vcross3(*e1, *e0));

    return 0;
}

template <int K = 7>
kr_internal kr_inline_device kr_error
lbvh_dito14_external_projection(cvec3* points, kr_size point_count, cvec3 normal, kr_scalar* min_proj, kr_scalar* max_proj)
{
    kr_scalar proj = kr_vdot3(points[0], normal);
    kr_scalar min_p = proj, max_p = proj;
    for (int i = 1; i < point_count; i++)
    {
        proj = kr_vdot3(points[i], normal);
        min_p = kr_min(min_p, proj);
        max_p = kr_max(max_p, proj);
    }

    *min_proj = min_p;
    *max_proj = max_p;

    return kr_success;
}

template <int K = 7>
kr_internal kr_inline_device void
lbvh_dito14_obb_from_normal_and_edges(
    cvec3* points, kr_size point_count,
    vec3 n,
    vec3 e0, vec3 e1, vec3 e2,
    vec3* b0, vec3* b1, vec3* b2, vec3* obb_len, vec3* obb_mid, kr_scalar* best_quality) {

    vec3 dmax, dmin, dlen;
    kr_scalar quality;

    cvec3 m0 = kr_vcross3(e0, n);
    cvec3 m1 = kr_vcross3(e1, n);
    cvec3 m2 = kr_vcross3(e2, n);

    // The operands are assumed to be orthogonal and unit normals	
    lbvh_dito14_external_projection(points, point_count, n, &dmin.y, &dmax.y);
    dlen.y = dmax.y - dmin.y;

    lbvh_dito14_external_projection(points, point_count, e0, &dmin.x, &dmax.x);
    dlen.x = dmax.x - dmin.x;

    lbvh_dito14_external_projection(points, point_count, m0, &dmin.z, &dmax.z);
    dlen.z = dmax.z - dmin.z;

    quality = lbvh_dito14_obb_quality(dlen);
    if (quality < *best_quality) { 
        *best_quality = quality; 
        *obb_mid = dmax;
        *obb_len = dmin;
        *b0 = e0; *b1 = n; *b2 = m0;
    }

    lbvh_dito14_external_projection(points, point_count, e1, &dmin.x, &dmax.x);
    dlen.x = dmax.x - dmin.x;

    lbvh_dito14_external_projection(points, point_count, m1, &dmin.z, &dmax.z);
    dlen.z = dmax.z - dmin.z;

    quality = lbvh_dito14_obb_quality(dlen);
    if (quality < *best_quality) { 
        *best_quality = quality; 
        *obb_mid = dmax;
        *obb_len = dmin;
        *b0 = e1; *b1 = n; *b2 = m1;
    }

    lbvh_dito14_external_projection(points, point_count, e2, &dmin.x, &dmax.x);
    dlen.x = dmax.x - dmin.x;

    lbvh_dito14_external_projection(points, point_count, m2, &dmin.z, &dmax.z);
    dlen.z = dmax.z - dmin.z;

    quality = lbvh_dito14_obb_quality(dlen);
    if (quality < *best_quality) { 
        *best_quality = quality; 
        *obb_mid = dmax;
        *obb_len = dmin;
        *b0 = e2; *b1 = n; *b2 = m2;
    }
}

template <int K = 7>
kr_internal kr_inline_device kr_error
lbvh_dito14_external_point(cvec3* points, kr_size point_count, cvec3 normal,
    kr_scalar* min_proj, kr_scalar* max_proj,
    vec3* min_proj_vert, vec3* max_proj_vert)
{
    kr_scalar proj = kr_vdot3(points[0], normal);
    kr_scalar min_p = proj, max_p = proj;
    vec3 min_pv = points[0], max_pv = points[0];

    for (int i = 1; i < point_count; i++)
    {
        proj = kr_vdot3(points[i], normal);
        if (proj < min_p) { min_p = proj; min_pv = points[i]; }
        if (proj > max_p) { max_p = proj; max_pv = points[i]; }
    }

    *min_proj = min_p;
    *max_proj = max_p;
    *min_proj_vert = min_pv;
    *max_proj_vert = max_pv;

    return kr_success;
}

template <int K = 7>
kr_internal kr_inline_device kr_error
lbvh_dito14_upper_lower_tetra_points(
    cvec3* points, kr_size point_count,
    vec3 n, vec3 p0, vec3 p1, vec3 p2, vec3* q0, vec3* q1, b32* q0valid, b32* q1valid) {
    kr_scalar max_proj, min_proj, tri_proj;
    kr_scalar eps = 0.000001f;

    *q0valid = kr_false;
    *q1valid = kr_false;

    lbvh_dito14_external_point(points, point_count, n, &min_proj, &max_proj, q1, q0);
    tri_proj = kr_vdot3(p0, n);

    if (max_proj - eps > tri_proj) { *q0valid = kr_true; }
    if (min_proj + eps < tri_proj) { *q1valid = kr_true; }

    return kr_success;
}

template <int K = 7>
kr_internal kr_inline_device i32
lbvh_dito14_upper_lower_tetra_construct(
    cvec3* points, kr_size point_count,
    vec3 n,
    vec3 p0, vec3 p1, vec3 p2,
    vec3 e0, vec3 e1, vec3 e2,
    vec3* b0, vec3* b1, vec3* b2, vec3* obb_len, vec3* obb_mid, kr_scalar* best_quality) {
    vec3 q0, q1;     // Top and bottom vertices for lower and upper tetra constructions
    vec3 f0, f1, f2; // Edge vectors towards q0;
    vec3 g0, g1, g2; // Edge vectors towards q1;
    vec3 n0, n1, n2; // Unit normals of top tetra tris
    vec3 m0, m1, m2; // Unit normals of bottom tetra tris

    b32 q0valid;
    b32 q1valid;

    lbvh_dito14_upper_lower_tetra_points(points, point_count, n, p0, p1, p2, &q0, &q1, &q0valid, &q1valid);

    if (!q0valid && !q1valid)
        return 1;

    if (q0valid) {
        f0 = kr_vnormalize3(kr_vsub3(q0, p0));
        f1 = kr_vnormalize3(kr_vsub3(q0, p1));
        f2 = kr_vnormalize3(kr_vsub3(q0, p2));
        n0 = kr_vnormalize3(kr_vcross3(f1, e0));
        n1 = kr_vnormalize3(kr_vcross3(f2, e1));
        n2 = kr_vnormalize3(kr_vcross3(f0, e2));
        lbvh_dito14_obb_from_normal_and_edges(points, point_count, n0, e0, f1, f0, b0, b1, b2, obb_len, obb_mid, best_quality);
        lbvh_dito14_obb_from_normal_and_edges(points, point_count, n1, e1, f2, f1, b0, b1, b2, obb_len, obb_mid, best_quality);
        lbvh_dito14_obb_from_normal_and_edges(points, point_count, n2, e2, f0, f2, b0, b1, b2, obb_len, obb_mid, best_quality);
    }

    if (q1valid) {
        g0 = kr_vnormalize3(kr_vsub3(q1, p0));
        g1 = kr_vnormalize3(kr_vsub3(q1, p1));
        g2 = kr_vnormalize3(kr_vsub3(q1, p2));
        m0 = kr_vnormalize3(kr_vcross3(g1, e0));
        m1 = kr_vnormalize3(kr_vcross3(g2, e1));
        m2 = kr_vnormalize3(kr_vcross3(g0, e2));
        lbvh_dito14_obb_from_normal_and_edges(points, point_count, m0, e0, g1, g0, b0, b1, b2, obb_len, obb_mid, best_quality);
        lbvh_dito14_obb_from_normal_and_edges(points, point_count, m1, e1, g2, g1, b0, b1, b2, obb_len, obb_mid, best_quality);
        lbvh_dito14_obb_from_normal_and_edges(points, point_count, m2, e2, g0, g2, b0, b1, b2, obb_len, obb_mid, best_quality);
    }

    return 0;
}

__global__ void
calculate_obbs_kernel_shared(
    const kr_object_cu* object, kr_bvh_node* nodes,
    const u32* parents, b32* visit_table,
    kr_obb3* obbs,
    kr_scalar* global_min_proj, kr_scalar* global_max_proj,
    kr_minmax_pair* global_minmax_pairs) {

    constexpr kr_u32 K = 7;
    extern __shared__ kr_scalar shared_mem[];

    __shared__ kr_scalar* shared_min_proj;
    __shared__ kr_scalar* shared_max_proj;
    __shared__ kr_minmax_pair* shared_minmax_proj;

    if (threadIdx.x == 0)
    {
        shared_min_proj = shared_mem;
        shared_max_proj = shared_min_proj + K * blockDim.x;
        shared_minmax_proj = (kr_minmax_pair*)(shared_max_proj + K * blockDim.x);
    }

    __syncthreads();

    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    const int firstThreadInBlock = blockIdx.x * blockDim.x;
    const int lastThreadInBlock = firstThreadInBlock + blockDim.x - 1;

    if (primitive_index >= primitive_count) return;

    kr_u32 current_index = (parents + internal_count)[primitive_index];
    kr_u32 lastNode_index = primitive_index + internal_count;

    kr_scalar local_min_proj[K];
    kr_scalar local_max_proj[K];
    kr_minmax_pair local_minmax_pairs[K];

    {
        kr_bvh_node* leaf = &((nodes + internal_count)[primitive_index]);
        u32 primitive_id = leaf->primitivesOffset;
        cuvec4 face = object->as_mesh.faces[primitive_id];
        cvec3 v0 = object->as_mesh.vertices[face.x];
        cvec3 v1 = object->as_mesh.vertices[face.y];
        cvec3 v2 = object->as_mesh.vertices[face.z];
        cvec3 points[] = { v0, v1, v2 };

        lbvh_dito14_fixed_projections(local_min_proj, local_max_proj, local_minmax_pairs, points, 3);
    }

    for (int ki = 0; ki < K; ++ki)
    {
        const size_t offset_ptr = K * (internal_count + primitive_index);
        const size_t shr_offset_ptr = K * threadIdx.x;
        (global_min_proj + offset_ptr)[ki] = local_min_proj[ki];
        (global_max_proj + offset_ptr)[ki] = local_max_proj[ki];
        (global_minmax_pairs + offset_ptr)[ki] = local_minmax_pairs[ki];
        (shared_min_proj + shr_offset_ptr)[ki] = local_min_proj[ki];
        (shared_max_proj + shr_offset_ptr)[ki] = local_max_proj[ki];
        (shared_minmax_proj + shr_offset_ptr)[ki] = local_minmax_pairs[ki];
    }

    __syncthreads();

    while (current_index != 0xFFFFFFFF) {
        b32 childThreadId = atomicExch(&visit_table[current_index], primitive_index);
        __threadfence();

        if (0xFFFFFFFF == childThreadId) { break; }

        kr_scalar* tmp_mem_min_proj;
        kr_scalar* tmp_mem_max_proj;
        kr_minmax_pair* tmp_mem_minmax_proj;

        if (childThreadId >= firstThreadInBlock &&
            childThreadId <= lastThreadInBlock)
        {
            const int childThreadIdInBlock = childThreadId - firstThreadInBlock;
            const size_t offset_ptr = K * childThreadIdInBlock;
            tmp_mem_min_proj = shared_min_proj + offset_ptr;
            tmp_mem_max_proj = shared_max_proj + offset_ptr;
            tmp_mem_minmax_proj = shared_minmax_proj + offset_ptr;
        }
        else
        {
            kr_bvh_node* current = &nodes[current_index];
            const int childIdx = current->left;
            const size_t offset_ptr = K * (childIdx == lastNode_index ? current->right : childIdx);
            tmp_mem_min_proj = global_min_proj + offset_ptr;
            tmp_mem_max_proj = global_max_proj + offset_ptr;
            tmp_mem_minmax_proj = global_minmax_pairs + offset_ptr;
        }

        for (int ki = 0; ki < K; ++ki)
        {
            if (local_min_proj[ki] > tmp_mem_min_proj[ki])
            {
                local_min_proj[ki] = tmp_mem_min_proj[ki];
                local_minmax_pairs[ki].min = tmp_mem_minmax_proj[ki].min;
            }

            if (local_max_proj[ki] < tmp_mem_max_proj[ki]) {
                local_max_proj[ki] = tmp_mem_max_proj[ki];
                local_minmax_pairs[ki].max = tmp_mem_minmax_proj[ki].max;
            }

            const size_t glb_offset_ptr = K * current_index;
            const size_t shared_offset_ptr = K * threadIdx.x;
            (global_min_proj + glb_offset_ptr)[ki] = local_min_proj[ki];
            (global_max_proj + glb_offset_ptr)[ki] = local_max_proj[ki];
            (global_minmax_pairs + glb_offset_ptr)[ki] = local_minmax_pairs[ki];
            (shared_min_proj + shared_offset_ptr)[ki] = local_min_proj[ki];
            (shared_max_proj + shared_offset_ptr)[ki] = local_max_proj[ki];
            (shared_minmax_proj + shared_offset_ptr)[ki] = local_minmax_pairs[ki];
        }

        __syncthreads();

        lastNode_index = current_index;
        current_index = parents[current_index];
    }
}

__global__ void
calculate_obbs_kernel(
    const kr_object_cu* object, kr_bvh_node* nodes, 
    const u32* parents, b32* visit_table,
    kr_obb3* obbs,
    kr_scalar* min_proj, kr_scalar* max_proj,
    kr_minmax_pair* minmax_pairs) {
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (primitive_index >= primitive_count)
        return;

    i32 i = primitive_index;

    kr_bvh_node* internal_nodes = nodes;
    kr_bvh_node* leaf_nodes = internal_nodes + internal_count;

    const u32* internal_parents = parents;
    const u32* leaf_parents = internal_parents + internal_count;

    kr_bvh_node* leaf = &leaf_nodes[i];
    kr_bvh_node* parent;

    kr_u32 parent_index = leaf_parents[i];

    while (parent_index != 0xFFFFFFFF) {
        parent = &internal_nodes[parent_index];

        b32 visited = atomicExch(&visit_table[parent_index], kr_true);
        __threadfence();
        if (kr_false == visited) {
            break;
        }

        constexpr auto K = 7;
        kr_scalar l_min_proj_mem[K];
        kr_scalar l_max_proj_mem[K];
        kr_minmax_pair l_minmax_pairs_mem[K];
        kr_scalar r_min_proj_mem[K];
        kr_scalar r_max_proj_mem[K];
        kr_minmax_pair r_minmax_pairs_mem[K];

        kr_scalar* l_min_proj = &l_min_proj_mem[0];
        kr_scalar* l_max_proj = &l_max_proj_mem[0];
        kr_minmax_pair* l_minmax_pairs = &l_minmax_pairs_mem[0];

        kr_scalar* r_min_proj = &r_min_proj_mem[0];
        kr_scalar* r_max_proj = &r_max_proj_mem[0];
        kr_minmax_pair* r_minmax_pairs = &r_minmax_pairs_mem[0];

        kr_scalar* o_min_proj = &min_proj[K * parent_index];
        kr_scalar* o_max_proj = &max_proj[K * parent_index];
        kr_minmax_pair* o_minmax_pairs = &minmax_pairs[K * parent_index];

        kr_bvh_node* l = &nodes[parent->left];
        kr_bvh_node* r = &nodes[parent->right];

        if (l->nPrimitives > 0) {
            u32 primitive_id = l->primitivesOffset;
            cuvec4 face = object->as_mesh.faces[primitive_id];
            cvec3 v0 = object->as_mesh.vertices[face.x];
            cvec3 v1 = object->as_mesh.vertices[face.y];
            cvec3 v2 = object->as_mesh.vertices[face.z];
            cvec3 points[] = { v0, v1, v2 };
            lbvh_dito14_fixed_projections(
                l_min_proj, l_max_proj, l_minmax_pairs,
                points, 3);
        }
        else {
            l_min_proj = &min_proj[K * parent->left];
            l_max_proj = &max_proj[K * parent->left];
            l_minmax_pairs = &minmax_pairs[K * parent->left];
        }

        if (r->nPrimitives > 0) {
            u32 primitive_id = r->primitivesOffset;
            cuvec4 face = object->as_mesh.faces[primitive_id];
            cvec3 v0 = object->as_mesh.vertices[face.x];
            cvec3 v1 = object->as_mesh.vertices[face.y];
            cvec3 v2 = object->as_mesh.vertices[face.z];
            cvec3 points[] = { v0, v1, v2 };
            lbvh_dito14_fixed_projections(
                r_min_proj, r_max_proj, r_minmax_pairs,
                points, 3);
        }
        else {
            r_min_proj = &min_proj[K * parent->right];
            r_max_proj = &max_proj[K * parent->right];
            r_minmax_pairs = &minmax_pairs[K * parent->right];
        }

        lbvh_dito14_fixed_projections_resolve(
            o_min_proj, o_max_proj, o_minmax_pairs,
            l_min_proj, l_max_proj, l_minmax_pairs,
            r_min_proj, r_max_proj, r_minmax_pairs
        );

        /*if (parent_index == 128249) {

            printf("Left %d\n", l->nPrimitives);
            printf("Right %d\n", r->nPrimitives);
            printf("GPU DiTO: Fixed Projections\n");
            for (int i = 0; i < 7; i++) {
                printf("GPU DiTO: Proj[%d] %f %f\n", i, o_min_proj[i], o_max_proj[i]);
            }
            printf("GPU DiTO: Fixed Points\n");
            for (int i = 0; i < 7; i++) {
                printf("GPU DiTO: Proj[%d] {%f %f %f} {%f %f %f}\n", i, o_minmax_pairs[i].min.x, o_minmax_pairs[i].min.y, o_minmax_pairs[i].min.z, o_minmax_pairs[i].max.x, o_minmax_pairs[i].max.y, o_minmax_pairs[i].max.z);
            }
        }*/

        parent_index = parents[parent_index];
    }
}

template <int K = 3>
kr_internal kr_inline_device kr_error
lbvh_dito14_axis_projections_resolve(
    kr_scalar* o_min_proj, kr_scalar* o_max_proj,
    const kr_scalar* l_min_proj, const kr_scalar* l_max_proj,
    const kr_scalar* r_min_proj, const kr_scalar* r_max_proj) {

    for (int i = 0; i < K; i++) {
        if (l_min_proj[i] < r_min_proj[i]) {
            o_min_proj[i] = l_min_proj[i];
        }
        else {
            o_min_proj[i] = r_min_proj[i];
        }
        if (l_max_proj[i] > r_max_proj[i]) {
            o_max_proj[i] = l_max_proj[i];
        }
        else {
            o_max_proj[i] = r_max_proj[i];
        }
    }

    return kr_success;
}


template <int K = 3>
kr_internal kr_inline_device kr_error
lbvh_dito14_axis_projections(
    kr_scalar* min_proj, kr_scalar* max_proj,
    const kr_vec3* points, kr_size point_count, 
    cvec3 b0, cvec3 b1, cvec3 b2
) {

    const auto& point = points[0];
    kr_scalar proj;
    vec3 obb_min;
    vec3 obb_max;

    proj = kr_vdot3(point, b0);
    obb_min.x = obb_max.x = proj;

    proj = kr_vdot3(point, b1);
    obb_min.y = obb_max.y = proj;

    proj = kr_vdot3(point, b2);
    obb_min.z = obb_max.z = proj;

    for (size_t i = 1; i < point_count; i++) {
        const auto& point = points[i];

        proj = kr_vdot3(point, b0);
        obb_min.x = kr_min(obb_min.x, proj);
        obb_max.x = kr_max(obb_max.x, proj);

        proj = kr_vdot3(point, b1);
        obb_min.y = kr_min(obb_min.y, proj);
        obb_max.y = kr_max(obb_max.y, proj);

        proj = kr_vdot3(point, b2);
        obb_min.z = kr_min(obb_min.z, proj);
        obb_max.z = kr_max(obb_max.z, proj);
    }

    min_proj[0] = obb_min.x;
    min_proj[1] = obb_min.y;
    min_proj[2] = obb_min.z;

    max_proj[0] = obb_max.x;
    max_proj[1] = obb_max.y;
    max_proj[2] = obb_max.z;

    return kr_success;
}

#if 0
template <typename F>
void lbvh_dito14_axis_projections(Vector<F>& mid, Vector<F>& len, OBB<F>& obb)
{
    obb.mid = mid;
    obb.ext = scalVecProd<F>(0.5, len);
    obb.v0 = createVector<F>(1, 0, 0);
    obb.v1 = createVector<F>(0, 1, 0);
    obb.v2 = createVector<F>(0, 0, 1);
}
#endif


__global__ void
finalize_obbs_kernel(
    const kr_object_cu* object, const kr_bvh_node* nodes,
    const kr_scalar* min_proj, const kr_scalar* max_proj,
    kr_obb3* obbs, kr_mat4* transforms
) {
    // TODO 

    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (node_index >= internal_count)
        return;

    kr_obb3* obb = &obbs[node_index];
    kr_mat4* transform = &transforms[node_index];
    const kr_scalar* o_min_proj = &min_proj[7 * node_index];
    const kr_scalar* o_max_proj = &max_proj[7 * node_index];
    
    cvec3 aabb_len = { o_max_proj[0] - o_min_proj[0], o_max_proj[1] - o_min_proj[1], o_max_proj[2] - o_min_proj[2] };
    cvec3 obb_len  = { o_max_proj[3] - o_min_proj[3], o_max_proj[4] - o_min_proj[4], o_max_proj[5] - o_min_proj[5] };

    kr_scalar aabb_quality = lbvh_dito14_obb_quality(aabb_len);
    kr_scalar obb_quality = lbvh_dito14_obb_quality(obb_len);

    if (obb_quality < aabb_quality) {
        cvec3 obb_min = { o_min_proj[3], o_min_proj[4], o_min_proj[5] };
        cvec3 obb_max = { o_max_proj[3], o_max_proj[4], o_max_proj[5] };

        cvec3 mid_lcs = kr_vmul31(kr_vadd3(obb_min, obb_max), 0.5f);
        obb->ext = kr_vmax3(kr_vmul31(kr_vsub3(obb_max, obb_min), 0.5f), kr_vof3(0.001f));
        
        obb->mid = kr_vmul31(obb->v0, mid_lcs.x);
        obb->mid = kr_vadd3(obb->mid, kr_vmul31(obb->v1, mid_lcs.y));
        obb->mid = kr_vadd3(obb->mid, kr_vmul31(obb->v2, mid_lcs.z));

        kr_mat4 r = kr_mobb3(*obb);
        kr_mat4 s = kr_mscale4({ obb->ext.x * 2.0f, obb->ext.y * 2.0f, obb->ext.z * 2.0f });
        kr_mat4 t = kr_mtranslate4(obb->mid);
        *transform = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));
    }
    else {
        cvec3 obb_min = { o_min_proj[0], o_min_proj[1], o_min_proj[2] };
        cvec3 obb_max = { o_max_proj[0], o_max_proj[1], o_max_proj[2] };

        obb->v0 = { 1, 0, 0 };
        obb->v1 = { 0, 1, 0 };
        obb->v2 = { 0, 0, 1 };

        cvec3 mid_lcs = kr_vmul31(kr_vadd3(obb_min, obb_max), 0.5f);
        obb->ext = kr_vmax3(kr_vmul31(kr_vsub3(obb_max, obb_min), 0.5f), kr_vof3(0.001f));

        obb->mid = kr_vmul31(obb->v0, mid_lcs.x);
        obb->mid = kr_vadd3(obb->mid, kr_vmul31(obb->v1, mid_lcs.y));
        obb->mid = kr_vadd3(obb->mid, kr_vmul31(obb->v2, mid_lcs.z));

        kr_mat4 r = kr_mobb3(*obb);
        kr_mat4 s = kr_mscale4({ obb->ext.x * 2.0f, obb->ext.y * 2.0f, obb->ext.z * 2.0f });
        kr_mat4 t = kr_mtranslate4(obb->mid);
        *transform = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));
    }
}

__global__ void
refit_obbs_kernel_new(
    const kr_object_cu* object, const kr_bvh_node* nodes,
    kr_obb3* obbs, kr_mat4* transforms, kr_obb_pair* obb_pairs,
    const u32* parents, const u32* primitive_counts,
    u32* processed_table, b32* visit_table,
    kr_scalar* min_proj, kr_scalar* max_proj
) {
    // TODO 
    //* switch to CUDA fminf/fmaxf
    //* investigate early exit options

    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (primitive_index >= primitive_count)
        return;

    i32 i = primitive_index;

    const kr_bvh_node* internal_nodes = nodes;
    const kr_bvh_node* leaf_nodes = internal_nodes + internal_count;

    const u32* internal_parents = parents;
    const u32* leaf_parents = internal_parents + internal_count;

    const u32* internal_counts = primitive_counts;
    const u32* leaf_counts = internal_counts + internal_count;

    const kr_bvh_node* leaf = &leaf_nodes[i];
    const kr_bvh_node* parent;

    kr_u32 parent_index = leaf_parents[i];

    u32 primitive_id = leaf->primitivesOffset;
    cuvec4 face = object->as_mesh.faces[primitive_id];
    cvec3 v0 = object->as_mesh.vertices[face.x];
    cvec3 v1 = object->as_mesh.vertices[face.y];
    cvec3 v2 = object->as_mesh.vertices[face.z];

    kr_obb3* obb = &obbs[internal_count + i];
    kr_mat4* transform = &transforms[internal_count + i];

    cvec3 e0 = kr_vsub3(v1, v0);
    cvec3 e1 = kr_vsub3(v2, v0);
    kr_scalar area = kr_vlength3(kr_vcross3(e0, e1)) * 0.5f;
    cvec3 a2 = kr_vcross3(e0, e1);
    cvec3 a0 = kr_vnormalize3(e0);
    cvec3 a1 = kr_vnormalize3(kr_vcross3(a0, a2));

    obb->v0 = a0;
    obb->v1 = a1;
    obb->v2 = a2;

    kr_mat4 r = kr_mobb3(*obb);

    cvec3 v0_lcs = kr_vtransform3(kr_minverse4(r), v0);
    cvec3 v1_lcs = kr_vtransform3(kr_minverse4(r), v1);
    cvec3 v2_lcs = kr_vtransform3(kr_minverse4(r), v2);

    vec3 max_lcs = v0_lcs;
    vec3 min_lcs = v0_lcs;

    max_lcs = kr_vmax3(max_lcs, kr_vmax3(v1_lcs, v2_lcs));
    min_lcs = kr_vmin3(min_lcs, kr_vmin3(v1_lcs, v2_lcs));

    cvec3 mid_lcs = kr_vmul31(kr_vadd3(max_lcs, min_lcs), 0.5f);
    obb->ext = kr_vmax3(kr_vmul31(kr_vsub3(max_lcs, min_lcs), 0.5f), kr_vof3(0.001f));
    obb->mid = kr_vmul31(obb->v0, mid_lcs.x);
    obb->mid = kr_vadd3(obb->mid, kr_vmul31(obb->v1, mid_lcs.y));
    obb->mid = kr_vadd3(obb->mid, kr_vmul31(obb->v2, mid_lcs.z));

    //obb->mid = kr_vtransform3(r, kr_vmul31(kr_vadd3(max_lcs, min_lcs), 0.5f));
    //obb->mid = kr_vmul31(kr_vadd3(max_lcs, min_lcs), 0.5f);

    //obb->ext = { kr_vlength3(e0), 1.0f / (2.0f * area * kr_vlength3(e0)), 0.0001f };
    //obb->ext = { 0.5f * kr_vlength3(e0), 0.5f * (2.0f * area / kr_vlength3(e0)), 0.001f };
    //obb->ext = { 11110.2f, 11110.2f, 11110.001f };

    kr_mat4 s = kr_mscale4({ obb->ext.x * 2.0f, obb->ext.y * 2.0f, obb->ext.z * 2.0f });
    kr_mat4 t = kr_mtranslate4(obb->mid);
    *transform = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));

    parent_index = leaf_parents[i];
    while (parent_index != 0xFFFFFFFF) {
        parent = &internal_nodes[parent_index];

        kr_obb3* obb = &obbs[parent_index];
        kr_obb_pair* obb_pair = &obb_pairs[parent_index];
        kr_scalar* o_min_proj = &min_proj[7 * parent_index];
        kr_scalar* o_max_proj = &max_proj[7 * parent_index];

        cvec3 proj_min = { o_min_proj[3], o_min_proj[4], o_min_proj[5] };
        cvec3 proj_max = { o_max_proj[3], o_max_proj[4], o_max_proj[5] };

        cvec3 b0 = obb->v0;
        cvec3 b1 = obb->v1;
        cvec3 b2 = obb->v2;

        kr_scalar proj;
        vec3 obb_min;
        vec3 obb_max;

        //Vertex 0
        proj = kr_vdot3(v0, b0);
        obb_min.x = obb_max.x = proj;

        proj = kr_vdot3(v0, b1);
        obb_min.y = obb_max.y = proj;

        proj = kr_vdot3(v0, b2);
        obb_min.z = obb_max.z = proj;

        //Vertex 1
        proj = kr_vdot3(v1, b0);
        obb_min.x = kr_min(obb_min.x, proj);
        obb_max.x = kr_max(obb_max.x, proj);

        proj = kr_vdot3(v1, b1);
        obb_min.y = kr_min(obb_min.y, proj);
        obb_max.y = kr_max(obb_max.y, proj);

        proj = kr_vdot3(v1, b2);
        obb_min.z = kr_min(obb_min.z, proj);
        obb_max.z = kr_max(obb_max.z, proj);

        //Vertex 2
        proj = kr_vdot3(v2, b0);
        obb_min.x = kr_min(obb_min.x, proj);
        obb_max.x = kr_max(obb_max.x, proj);

        proj = kr_vdot3(v2, b1);
        obb_min.y = kr_min(obb_min.y, proj);
        obb_max.y = kr_max(obb_max.y, proj);

        proj = kr_vdot3(v2, b2);
        obb_min.z = kr_min(obb_min.z, proj);
        obb_max.z = kr_max(obb_max.z, proj);

        if (obb_min.x < proj_min.x) atomicMin(&o_min_proj[3], obb_min.x);
        if (obb_min.y < proj_min.y) atomicMin(&o_min_proj[4], obb_min.y);
        if (obb_min.z < proj_min.z) atomicMin(&o_min_proj[5], obb_min.z);

        if (obb_max.x > proj_max.x) atomicMax(&o_max_proj[3], obb_max.x);
        if (obb_max.y > proj_max.y) atomicMax(&o_max_proj[4], obb_max.y);
        if (obb_max.z > proj_max.z) atomicMax(&o_max_proj[5], obb_max.z);

        parent_index = internal_parents[parent_index];
    }
}

__global__ void
refit_obbs_kernel(
    const kr_object_cu* object, const kr_bvh_node* nodes,
    kr_obb3* obbs,
    const u32* parents, const u32* primitive_counts,
    u32* processed_table, b32* visit_table,
    kr_scalar* min_proj, kr_scalar* max_proj
) {
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (primitive_index >= primitive_count)
        return;
    b32 root_processed = (b32)atomicAdd(&processed_table[0], 0);
    if (root_processed) return;

    i32 i = primitive_index;

    const kr_bvh_node* internal_nodes = nodes;
    const kr_bvh_node* leaf_nodes = internal_nodes + internal_count;

    const u32* internal_parents = parents;
    const u32* leaf_parents = internal_parents + internal_count;

    const u32* internal_counts = primitive_counts;
    const u32* leaf_counts = internal_counts + internal_count;

    const kr_bvh_node* leaf = &leaf_nodes[i];
    const kr_bvh_node* parent;

    kr_u32 parent_index = leaf_parents[i];

    u32 stop_index = 0;
    parent_index = leaf_parents[i];
    vec3 b0, b1, b2;
    while (parent_index != 0xFFFFFFFF) {
        parent = &internal_nodes[parent_index];
        b32 processed = (b32)atomicAdd(&processed_table[parent_index], 0);
        if (!processed) {
            kr_obb3* obb = &obbs[parent_index];

            b0 = obb->v0;
            b1 = obb->v1;
            b2 = obb->v2;

            stop_index = parent_index;
            break;
        }

        parent_index = internal_parents[parent_index];
    }

    if (parent_index == 0xFFFFFFFF) return;

    parent_index = leaf_parents[i];

    while (kr_true) {
        parent = &internal_nodes[parent_index];
        b32 visited = atomicExch(&visit_table[parent_index], kr_true);
        __threadfence();
        if (kr_false == visited) {
            break;
        }
        atomicExch((kr_u32*)&processed_table[parent_index], kr_true);

        kr_obb3* obb = &obbs[parent_index];

        constexpr auto K = 3;
        kr_scalar l_min_proj_mem[K];
        kr_scalar l_max_proj_mem[K];
        kr_scalar r_min_proj_mem[K];
        kr_scalar r_max_proj_mem[K];

        kr_scalar* l_min_proj = &l_min_proj_mem[0];
        kr_scalar* l_max_proj = &l_max_proj_mem[0];

        kr_scalar* r_min_proj = &r_min_proj_mem[0];
        kr_scalar* r_max_proj = &r_max_proj_mem[0];

        kr_scalar* o_min_proj = &min_proj[K * parent_index];
        kr_scalar* o_max_proj = &max_proj[K * parent_index];

        const kr_bvh_node* l = &nodes[parent->left];
        const kr_bvh_node* r = &nodes[parent->right];

        if (l->nPrimitives > 0) {
            u32 primitive_id = l->primitivesOffset;
            cuvec4 face = object->as_mesh.faces[primitive_id];
            cvec3 v0 = object->as_mesh.vertices[face.x];
            cvec3 v1 = object->as_mesh.vertices[face.y];
            cvec3 v2 = object->as_mesh.vertices[face.z];
            cvec3 points[] = { v0, v1, v2 };
            lbvh_dito14_axis_projections(
                l_min_proj, l_max_proj,
                points, 3, b0, b1, b2);
        }
        else {
            l_min_proj = &min_proj[K * parent->left];
            l_max_proj = &max_proj[K * parent->left];
        }

        if (r->nPrimitives > 0) {
            u32 primitive_id = r->primitivesOffset;
            cuvec4 face = object->as_mesh.faces[primitive_id];
            cvec3 v0 = object->as_mesh.vertices[face.x];
            cvec3 v1 = object->as_mesh.vertices[face.y];
            cvec3 v2 = object->as_mesh.vertices[face.z];
            cvec3 points[] = { v0, v1, v2 };
            lbvh_dito14_axis_projections(
                r_min_proj, r_max_proj,
                points, 3, b0, b1, b2);
        }
        else {
            r_min_proj = &min_proj[K * parent->right];
            r_max_proj = &max_proj[K * parent->right];
        }

        lbvh_dito14_axis_projections_resolve(
            o_min_proj, o_max_proj,
            l_min_proj, l_max_proj,
            r_min_proj, r_max_proj
        );

        if (parent_index == stop_index) {
            cvec3* obb_min = (cvec3*)o_min_proj;
            cvec3* obb_max = (cvec3*)o_max_proj;
            obb->ext = kr_vmul31(kr_vsub3(*obb_max, *obb_min), 0.5f);
           
            kr_scalar quality_obb = 4 * lbvh_dito14_obb_quality(obb->ext);
            kr_scalar quality_aabb = lbvh_dito14_obb_quality(kr_aabb_extents3(parent->bounds));
            if (parent_index == 0) {
                printf("Qualities %f %f\n", quality_obb, quality_aabb);
            }

            if (quality_aabb < quality_obb) {
                b0 = { 1, 0, 0 };
                b1 = { 0, 1, 0 };
                b2 = { 0, 0, 1 };

                cvec3 mid_lcs = kr_aabb_center3(parent->bounds);
                obb->mid = kr_vmul31(b0, mid_lcs.x);
                obb->mid = kr_vadd3(obb->mid, kr_vmul31(b1, mid_lcs.y));
                obb->mid = kr_vadd3(obb->mid, kr_vmul31(b2, mid_lcs.z));
                obb->ext = kr_aabb_extents3(parent->bounds);
                obb->ext = kr_vmul31(obb->ext, 0.5f);
                obb->v0 = b0;
                obb->v1 = b1;
                obb->v2 = b2;
            }
            else {
                cvec3 mid_lcs = kr_vmul31(kr_vadd3(*obb_max, *obb_min), 0.5f);
                obb->mid = kr_vmul31(b0, mid_lcs.x);
                obb->mid = kr_vadd3(obb->mid, kr_vmul31(b1, mid_lcs.y));
                obb->mid = kr_vadd3(obb->mid, kr_vmul31(b2, mid_lcs.z));
                obb->v0 = b0;
                obb->v1 = b1;
                obb->v2 = b2;
            }

            break;
        }
        parent_index = internal_parents[parent_index];
    }
}

__global__ void
calculate_candidate_obb_kernel(
    const kr_object_cu* object, 
    kr_obb3* obbs, kr_obb_pair* obb_pairs,
    kr_scalar* min_proj, kr_scalar* max_proj,
    const kr_minmax_pair* minmax_pairs) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (node_index >= internal_count)
        return;
    //if (node_index != 128249)
    //    return;

    kr_obb3* obb = &obbs[node_index];

    constexpr auto K = 7;
    const kr_minmax_pair* o_minmax_pairs = &minmax_pairs[K * node_index];
    kr_scalar* o_min_proj = &min_proj[K * node_index];
    kr_scalar* o_max_proj = &max_proj[K * node_index];

    kr_minmax_pair l_minmax_pairs[K] = {
        o_minmax_pairs[0], o_minmax_pairs[1], o_minmax_pairs[2], o_minmax_pairs[3],
        o_minmax_pairs[4], o_minmax_pairs[5], o_minmax_pairs[6]
    };

    vec3 aabb_mid = { (o_min_proj[0] + o_max_proj[0]) * 0.5f, (o_min_proj[1] + o_max_proj[1]) * 0.5f, (o_min_proj[2] + o_max_proj[2]) * 0.5f };
    vec3 aabb_len = { o_max_proj[0] - o_min_proj[0], o_max_proj[1] - o_min_proj[1], o_max_proj[2] - o_min_proj[2] };
    kr_scalar aabb_quality = lbvh_dito14_obb_quality(aabb_len);

    vec3 b0 = { 1, 0, 0 };
    vec3 b1 = { 0, 1, 0 };
    vec3 b2 = { 0, 0, 1 };
    vec3 n, p0, p1, p2, e0, e1, e2;
    vec3 obb_len = { o_min_proj[0], o_min_proj[1], o_min_proj[2] };
    vec3 obb_mid = { o_max_proj[0], o_max_proj[1], o_max_proj[2] };
    kr_scalar best_quality = aabb_quality;

    i32 ret = lbvh_dito14_base_triangle_construct(
        l_minmax_pairs, (cvec3*)&l_minmax_pairs[0], 2 * K,
        &n,
        &p0, &p1, &p2,
        &e0, &e1, &e2
    );

    if (ret != 0) {
        obb->v0 = b0;
        obb->v1 = b1;
        obb->v2 = b2;

        return;
    }

    lbvh_dito14_obb_from_normal_and_edges(
        (cvec3*)&l_minmax_pairs[0], 2 * K, n, e0, e1, e2, &b0, &b1, &b2, &obb_len, &obb_mid, &best_quality
    );

    if (node_index == 0) {
        printf("GPU DiTO Best Value %f %d\n", best_quality, ret);
        printf("GPU DiTO p0 %f %f %f\n", p0.x, p0.y, p0.z);
        printf("GPU DiTO p1 %f %f %f\n", p1.x, p1.y, p1.z);
        printf("GPU DiTO p2 %f %f %f\n", p2.x, p2.y, p2.z);
        printf("GPU DiTO e0 %f %f %f\n", e0.x, e0.y, e0.z);
        printf("GPU DiTO e1 %f %f %f\n", e1.x, e1.y, e1.z);
        printf("GPU DiTO e2 %f %f %f\n", e2.x, e2.y, e2.z);
        printf("GPU DiTO b0 %f %f %f\n", b0.x, b0.y, b0.z);
        printf("GPU DiTO b1 %f %f %f\n", b1.x, b1.y, b1.z);
        printf("GPU DiTO b2 %f %f %f\n", b2.x, b2.y, b2.z);
    }

    ret = lbvh_dito14_upper_lower_tetra_construct(
        (cvec3*)&l_minmax_pairs[0], 2 * K,
        n,
        p0, p1, p2,
        e0, e1, e2,
        &b0, &b1, &b2, &obb_len, &obb_mid, &best_quality
    );

    if (ret != 0) {
        obb->v0 = { 1, 0, 0 };
        obb->v1 = { 0, 1, 0 };
        obb->v2 = { 0, 0, 1 };

        return;
    }

    //obb_len = kr_vmul31(obb_len, 0.5f);

    if (node_index == 0) {
        printf("GPU DiTO Best Value %f %d\n", best_quality, ret);
        printf("GPU DiTO AL   Value %f \n", aabb_quality);
        printf("GPU DiTO p0 %f %f %f\n", p0.x, p0.y, p0.z);
        printf("GPU DiTO p1 %f %f %f\n", p1.x, p1.y, p1.z);
        printf("GPU DiTO p2 %f %f %f\n", p2.x, p2.y, p2.z);
        printf("GPU DiTO e0 %f %f %f\n", e0.x, e0.y, e0.z);
        printf("GPU DiTO e1 %f %f %f\n", e1.x, e1.y, e1.z);
        printf("GPU DiTO e2 %f %f %f\n", e2.x, e2.y, e2.z);
        printf("GPU DiTO b0 %f %f %f\n", b0.x, b0.y, b0.z);
        printf("GPU DiTO b1 %f %f %f\n", b1.x, b1.y, b1.z);
        printf("GPU DiTO b2 %f %f %f\n", b2.x, b2.y, b2.z);
    }

    obb->v0 = b0;
    obb->v1 = b1;
    obb->v2 = b2;

    o_min_proj[3] = obb_len.x;
    o_min_proj[4] = obb_len.y;
    o_min_proj[5] = obb_len.z;

    o_max_proj[3] = obb_mid.x;
    o_max_proj[4] = obb_mid.y;
    o_max_proj[5] = obb_mid.z;
}

__global__ void
calculate_leaf_obbs_kernel(
    const kr_object_cu* object, kr_bvh_node* nodes,
    u32* parents, b32* visit_table,
    kr_obb3* obbs,
    kr_scalar* min_proj, kr_scalar* max_proj,
    kr_minmax_pair* minmax_pairs) {
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    if (primitive_index >= primitive_count)
        return;
#if 0
    i32 i = primitive_index;

    kr_bvh_node* internal_nodes = nodes;
    kr_bvh_node* leaf_nodes = internal_nodes + internal_count;

    u32* internal_parents = parents;
    u32* leaf_parents = internal_parents + internal_count;

    kr_bvh_node* leaf = &leaf_nodes[i];
    kr_bvh_node* parent;

    kr_u32 parent_index = leaf_parents[i];
    parent = &internal_nodes[parent_index];

    b32 visited = atomicExch(&visit_table[parent_index], kr_true);
    if (kr_false == visited) {
        break;
    }

    kr_obb3* obb = &obbs[parent_index];

    constexpr auto K = 7;
    kr_scalar l_min_proj_mem[K];
    kr_scalar l_max_proj_mem[K];
    kr_minmax_pair l_minmax_pairs_mem[K];
    kr_scalar r_min_proj_mem[K];
    kr_scalar r_max_proj_mem[K];
    kr_minmax_pair r_minmax_pairs_mem[K];

    kr_scalar* l_min_proj = &l_min_proj_mem[0];
    kr_scalar* l_max_proj = &l_max_proj_mem[0];
    kr_minmax_pair* l_minmax_pairs = &l_minmax_pairs_mem[0];

    kr_scalar* r_min_proj = &r_min_proj_mem[0];
    kr_scalar* r_max_proj = &r_max_proj_mem[0];
    kr_minmax_pair* r_minmax_pairs = &r_minmax_pairs_mem[0];

    kr_scalar* o_min_proj = &min_proj[K * parent_index];
    kr_scalar* o_max_proj = &max_proj[K * parent_index];
    kr_minmax_pair* o_minmax_pairs = &minmax_pairs[K * parent_index];

    kr_bvh_node* l = &nodes[parent->left];
    kr_bvh_node* r = &nodes[parent->right];


    u32 primitive_id = l->primitivesOffset;
    cuvec4 face = object->as_mesh.faces[primitive_id];
    cvec3 v0 = object->as_mesh.vertices[face.x];
    cvec3 v1 = object->as_mesh.vertices[face.y];
    cvec3 v2 = object->as_mesh.vertices[face.z];
    cvec3 points[] = { v0, v1, v2 };
    lbvh_dito14_fixed_projections(
        l_min_proj, l_max_proj, l_minmax_pairs,
        points, 3);
    

    u32 primitive_id = r->primitivesOffset;
    cuvec4 face = object->as_mesh.faces[primitive_id];
    cvec3 v0 = object->as_mesh.vertices[face.x];
    cvec3 v1 = object->as_mesh.vertices[face.y];
    cvec3 v2 = object->as_mesh.vertices[face.z];
    cvec3 points[] = { v0, v1, v2 };
    lbvh_dito14_fixed_projections(
        r_min_proj, r_max_proj, r_minmax_pairs,
        points, 3);
  

    lbvh_dito14_fixed_projections_resolve(
        o_min_proj, o_max_proj, o_minmax_pairs,
        l_min_proj, l_max_proj, l_minmax_pairs,
        r_min_proj, r_max_proj, r_minmax_pairs
    );

    vec3 aabb_mid = { (o_min_proj[0] + o_max_proj[0]) * 0.5f, (o_min_proj[1] + o_max_proj[1]) * 0.5f, (o_min_proj[2] + o_max_proj[2]) * 0.5f };
    vec3 aabb_len = { o_max_proj[0] - o_min_proj[0], o_max_proj[1] - o_min_proj[1], o_max_proj[2] - o_min_proj[2] };
    kr_scalar aabb_quality = lbvh_dito14_obb_quality(aabb_len);

    vec3 b0 = { 1, 0, 0 };
    vec3 b1 = { 0, 1, 0 };
    vec3 b2 = { 0, 0, 1 };
    vec3 n, p0, p1, p2, e0, e1, e2;
    kr_scalar best_quality = aabb_quality;
    lbvh_dito14_base_triangle_construct(
        o_minmax_pairs, (cvec3*)o_minmax_pairs, 14,
        &n,
        &p0, &p1, &p2,
        &e0, &e1, &e2
    );
#endif
}


kr_internal kr_error
lbvh_cuda_calculate_obbs_top_down(kr_ads_lbvh_cuda* ads) {
    kr_scene* scene = ads->scene;
    cudaError_t cu_error;

    kr_ads_blas_cuda* blas = &ads->blas;
    kr_bvh_node* d_bvh = blas->bvh;

    KR_ALLOC_DECLARE(kr_obb3, h_obbs, blas->node_count);
    KR_ALLOC_DECLARE(kr_mat4, h_transforms, blas->node_count);
    KR_ALLOC_DECLARE(u32, h_counts, blas->node_count);
    KR_ALLOC_DECLARE(u32, h_parents, blas->node_count);
    KR_ALLOC_DECLARE(b32, visit_table, blas->internal_count);
    KR_ALLOC_DECLARE(kr_bvh_node, h_bvh, blas->node_count);
    
    cu_error = cudaMemcpy(h_bvh, d_bvh, blas->node_count * sizeof(*h_bvh), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_parents, blas->parents, blas->node_count * sizeof(*h_parents), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_counts, blas->primitive_counts, blas->node_count * sizeof(*h_counts), cudaMemcpyDeviceToHost);

    kr_bvh_node* nodes = h_bvh;
    kr_bvh_node* leaf_nodes = nodes + blas->internal_count;
    kr_bvh_node* internal_nodes = nodes;

    kr_mat4* transforms = h_transforms;
    kr_mat4* leaf_transforms = transforms + blas->internal_count;
    kr_mat4* node_transforms = transforms;

    u32* leaf_parents = h_parents + blas->internal_count;
    u32* internal_parents = h_parents;

    u32* counts = h_counts;
    u32* leaf_counts = counts + blas->internal_count;
    u32* internal_counts = counts;

    std::vector<vec3> vertices(blas->primitive_count * 3);
    kr_object* object = &ads->scene->objects[0];

    for (u32 i = 0; i < blas->primitive_count; ++i) {
        kr_bvh_node* leaf = &leaf_nodes[i];

        u32 primitive_id = (leaf->nPrimitives == 1) ? leaf->primitivesOffset : blas->primitives[leaf->primitivesOffset + i];

        cuvec4 face = object->as_mesh.faces[primitive_id];
        vertices[0] = object->as_mesh.vertices[face.x];
        vertices[1] = object->as_mesh.vertices[face.y];
        vertices[2] = object->as_mesh.vertices[face.z];

        kr_obb3 obb = kr_points_obb(vertices.data(), 3, nullptr);

        obb.ext = kr_vmax3(obb.ext, kr_vof3(0.001f));
        kr_mat4 r = kr_mobb3(obb);
        kr_mat4 s = kr_mscale4(KR_INITIALIZER_CAST(vec3) { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f });
        kr_mat4 t = kr_mtranslate4(obb.mid);
        kr_mat4 trs = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));
        h_obbs[blas->internal_count + i] = obb;
        leaf_transforms[i] = trs;
        leaf->axis = 1;
#if 1
        kr_bvh_node* parent;
        kr_u32 parent_index = leaf_parents[i];
        while (parent_index != 0xFFFFFFFF) {
            parent = &internal_nodes[parent_index];
            /*if (parent_index != 128249) {
                parent_index = internal_parents[parent_index];
                continue;
            }*/

            if (kr_false == kr_atomic_cmp_exch((kr_u32*)&visit_table[parent_index], kr_true, kr_false)) {
                break;
            }

            aabb3 aabb = parent->bounds;
            u32   count = counts[parent_index];
            //if (count > 1024)
            //    break;

            const auto& leaves = lbvh_node_leaves(nodes, parent);
            i32 vertex_count = 0;
            for (const auto& leaf : leaves) {
                u32 primitive_id = (leaf->nPrimitives == 1) ? leaf->primitivesOffset : blas->primitives[leaf->primitivesOffset + i];

                cuvec4 face = object->as_mesh.faces[primitive_id];
                vertices[vertex_count + 0] = object->as_mesh.vertices[face.x];
                vertices[vertex_count + 1] = object->as_mesh.vertices[face.y];
                vertices[vertex_count + 2] = object->as_mesh.vertices[face.z];
                vertex_count += 3;
            }

            /*if (parent_index == 120300) {
                printf("Stop 120300\n");
            }
            if (parent_index == 128249) {
                printf("Stop 128249\n");
            }*/
            kr_obb3 obb = kr_points_obb(vertices.data(), vertex_count, nullptr);
            obb.ext = kr_vmax3(obb.ext, kr_vof3(0.001f));
            kr_mat4 r = kr_mobb3(obb);
            kr_mat4 s = kr_mscale4(KR_INITIALIZER_CAST(vec3) { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f });
            kr_mat4 t = kr_mtranslate4(obb.mid);
            kr_mat4 trs = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));

            h_obbs[parent_index] = obb;
            //ads->obbs_brute_force[parent_index] = obb;
            node_transforms[parent_index] = trs;
            parent->axis = 1;
            parent_index = internal_parents[parent_index];
        }
#endif
    }
#if 0
    kr_mat4 km = kr_mfrom16(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
    glm::mat4 gm = glm::mat4(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
    glm::mat4 gt = glm::translate(glm::mat4(1.0f), { 1.0f,2.0f,3.0f });
    kr_mat4 kt = kr_mtranslate4({ 1.0f,2.0f,3.0f });
    glm::mat4 gr = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), { 0.0f, 1.0f, 0.0f });
    kr_mat4 kr = kr_mrotate4({ 0.0f, 1.0f, 0.0f }, kr_radians(45.0f));
   
    glm::quat qr = glm::quat_cast(gr);
    
    printf("testing\n");

    kr_scalar step = 5.0f;
    kr_scalar start = -10.0f;
    kr_scalar end = 10.0f;
    for (kr_scalar x = start; x <= end; x += step) {
        for (kr_scalar y = start; y <= end; y += step) {
            for (kr_scalar z = start; z <= end; z += step) {
                glm::vec3 a = { 0.0f, 1.0f, 0.0f };
                kr_scalar an = glm::radians(45.0f);
                vec3 p = { x, y, z };
                glm::vec3 ap = { x, y, z };
                vec3 akp = { x, y, z };
                vec3 kp = { x, y, z };
                glm::vec3 gp = { x, y, z };

                glm::mat4 gr = glm::rotate(glm::mat4(1.0f), an, a);
                kr_mat4 kr = kr_mrotate4({ a.x, a.y, a.z }, an);

                gp = glm::vec3(gr * glm::vec4(gp, 1.0f));
                kp = kr_vtransform3(kr, kp);

                ap = kr_cos(an) * ap
                    + kr_sin(an) * glm::cross(a, ap)
                    + (1.0f - kr_cos(an)) * glm::dot(a, ap) * a;

                akp = kr_vrotate_angle_axis3(akp, { a.x, a.y, a.z }, an);

                printf("{%f %f %f} to \n"
                       "               {%f %f %f}\n"
                       "               {%f %f %f}\n"
                       "               {%f %f %f}\n"
                       "               {%f %f %f}\n",
                    p.x, p.y, p.z, 
                    kp.x, kp.y, kp.z,
                    gp.x, gp.y, gp.z,
                    ap.x, ap.y, ap.z,
                    akp.x, akp.y, akp.z);
            }
        }
    }

    step = 0.5f;
    start = -1.0f;
     end = 1.0f;
    for (kr_scalar x = start; x <= end; x += step) {
        for (kr_scalar y = start; y <= end; y += step) {
            for (kr_scalar z = start; z <= end; z += step) {
                kr_vec3 a = kr_vnormalize3({ x, y, z });
                kr_scalar an = glm::radians(15.0f);

                kr_mat4 kt = kr_mrotate4(a, an);
                //glm::mat4 gt = glm::rotate(glm::mat4(1.0f), an, { a.x, a.y, a.z });
                vec4 kaa = kr_mangle_axis4(kt);
                glm::mat4 gt = glm::mat4_cast(glm::angleAxis(kaa.w, glm::vec3{ kaa.x, kaa.y, kaa.z }));

                bool should_print = false;
                kr_scalar epsilon = 0.001f;
                for (i32 i = 0; i < 16; i++) {
                    if (kr_abs(glm::value_ptr(gt)[i] - kt.v[i]) > epsilon) {
                        should_print = true;
                    }
                }

                if (!should_print) 
                    continue;

                printf("{%f %f %f %f} to \n"
                    "   {%f %f %f %f} {%f %f %f %f}\n"
                    "   {%f %f %f %f} {%f %f %f %f}\n"
                    "   {%f %f %f %f} {%f %f %f %f}\n"
                    "   {%f %f %f %f} {%f %f %f %f}\n",
                    kaa.x, kaa.y, kaa.z, kaa.w,
                    kt.c00, kt.c01, kt.c02, kt.c03, gt[0][0], gt[0][1], gt[0][2], gt[0][3],
                    kt.c10, kt.c11, kt.c12, kt.c13, gt[1][0], gt[1][1], gt[1][2], gt[1][3],
                    kt.c20, kt.c21, kt.c22, kt.c23, gt[2][0], gt[2][1], gt[2][2], gt[2][3],
                    kt.c30, kt.c31, kt.c32, kt.c33, gt[3][0], gt[3][1], gt[3][2], gt[3][3]);
            }
        }
    }
#endif
    cu_error = cudaMemcpy(d_bvh, h_bvh, blas->node_count * sizeof(*h_bvh), cudaMemcpyHostToDevice);
    cu_error = cudaMemcpy(blas->obb_transforms, h_transforms, blas->node_count * sizeof(*h_transforms), cudaMemcpyHostToDevice);
    cu_error = cudaMemcpy(blas->obbs, h_obbs, blas->node_count * sizeof(*h_obbs), cudaMemcpyHostToDevice);
    //kr_bvh_nodes_export(h_bvh, h_bvh, kr_null, 19, "tree_aabbs");
    //kr_bvh_nodes_export(h_bvh, h_bvh, h_transforms, 19, "tree_obbs");
    //exit(0);
#if 0
    for (kr_size i = 0; i < ads->internal_count; i++) {
        kr_obb3* ours = &ads->obbs[i];
        kr_obb3* ours_slow = &ads->obbs_brute_force[i];
        kr_obb3 obb = *ours;

        kr_obb3 obb2 = *ours_slow;

        if (!kr_vequal3(obb.mid, obb2.mid)) {
            printf("\n\n\n");

            printf("OURS[%d]: OBB Mid %f %f %f\n", i, obb.mid.x, obb.mid.y, obb.mid.z);
            printf("OURS[%d]: OBB Ext %f %f %f\n", i, obb.ext.x, obb.ext.y, obb.ext.z);
            printf("OURS[%d]: OBB A0  %f %f %f\n", i, obb.v0.x, obb.v0.y, obb.v0.z);
            printf("OURS[%d]: OBB A1  %f %f %f\n", i, obb.v1.x, obb.v1.y, obb.v1.z);
            printf("OURS[%d]: OBB A2  %f %f %f\n", i, obb.v2.x, obb.v2.y, obb.v2.z);
            printf("------------------------\n");

            printf("OURS[%d]: OBB Mid %f %f %f\n", i, obb2.mid.x, obb2.mid.y, obb2.mid.z);
            printf("OURS[%d]: OBB Ext %f %f %f\n", i, obb2.ext.x, obb2.ext.y, obb2.ext.z);
            printf("OURS[%d]: OBB A0  %f %f %f\n", i, obb2.v0.x, obb2.v0.y, obb2.v0.z);
            printf("OURS[%d]: OBB A1  %f %f %f\n", i, obb2.v1.x, obb2.v1.y, obb2.v1.z);
            printf("OURS[%d]: OBB A2  %f %f %f\n", i, obb2.v2.x, obb2.v2.y, obb2.v2.z);
        }
    }
#endif

    return kr_success;
}

kr_internal kr_error
lbvh_cuda_calculate_obbs(kr_ads_lbvh_cuda* ads, kr_ads_obb_metrics * metrics) {
    //return lbvh_cuda_calculate_obbs_top_down(ads);
    
    cudaError_t cu_error;
    kr_ads_blas_cuda* blas = &ads->blas;
    kr_object_cu* object_cu = blas->d_object_cu;
    const kr_object_cu* object = blas->h_object_cu;
    kr_scalar elapsed_ms = 0.0f;
    
    return kr_cuda_bvh_obb_tree(
        blas->bvh, blas->parents,
        blas->primitive_counts, blas->costs,
        object->as_mesh.vertices,
        object->as_mesh.faces,
        blas->primitives,
        blas->obbs,
        blas->obb_transforms,
        ads->cost_internal, ads->cost_traversal, 1.0f,
        blas->leaf_count, blas->internal_count, -1
    );


    kr_u32 primitive_count = blas->primitive_count;
    kr_u32 internal_count  = blas->internal_count;

    constexpr auto K = 7;

    KR_ALLOC_DECLARE(kr_bvh_node, h_bvh, blas->node_count);
    KR_ALLOC_DECLARE(kr_scalar, h_min_proj, K * blas->node_count);
    KR_ALLOC_DECLARE(kr_scalar, h_max_proj, K * blas->node_count);
    KR_ALLOC_DECLARE(kr_minmax_pair, h_minmax_pairs, K * blas->node_count);
    KR_ALLOC_DECLARE(kr_obb3, h_obbs, blas->node_count);
    KR_ALLOC_DECLARE(kr_mat4, h_trasnfroms, blas->node_count);

    //blas->min_proj = (kr_scalar*)kr_cuda_allocate(K * blas->internal_count * sizeof(*blas->min_proj));
    //blas->max_proj = (kr_scalar*)kr_cuda_allocate(K * blas->internal_count * sizeof(*blas->max_proj));
    //blas->minmax_pairs = (kr_minmax_pair*)kr_cuda_allocate(K * blas->internal_count * sizeof(*blas->minmax_pairs));

    /*kr_cuda_free((void**)&blas->transformations);
    kr_cuda_free((void**)&blas->bvh_packed);
    kr_cuda_free((void**)&blas->ray_counter);
    kr_cuda_free((void**)&blas->collapsed_bvh);
    kr_cuda_free((void**)&blas->costs);
    kr_cuda_free((void**)&blas->primitive_counts);
    kr_cuda_free((void**)&blas->mortons);*/
    //kr_cuda_free((void**)&blas->obb_transforms);
    
    blas->min_proj = (kr_scalar*)kr_cuda_allocate(K * blas->node_count * sizeof(*blas->min_proj));
    blas->max_proj = (kr_scalar*)kr_cuda_allocate(K * blas->node_count * sizeof(*blas->max_proj));
    blas->minmax_pairs = (kr_minmax_pair*)kr_cuda_allocate(K * blas->node_count * sizeof(*blas->minmax_pairs));
    
    printf("%p %p %p\n", blas->min_proj, blas->max_proj, blas->minmax_pairs);

    //kr_obb_pair* obb_pairs = (kr_obb_pair*)kr_cuda_allocate(blas->internal_count * sizeof(*obb_pairs));

    //u32* processed_table = (u32*)kr_cuda_allocate(blas->internal_count * sizeof(*processed_table));

    cu_error = cudaMemset(blas->visit_table, 0, blas->internal_count * sizeof(*blas->visit_table));

#define opt_kernel

    for (int i = 0; i < 1; i++) {
#ifdef opt_kernel
        cu_error = cudaMemset(blas->visit_table, 0xFFFFFFFF, blas->internal_count * sizeof(*blas->visit_table));
#else
        cu_error = cudaMemset(blas->visit_table, 0, blas->internal_count * sizeof(*blas->visit_table));
#endif
        //cu_error = cudaMemset(processed_table, 0, blas->internal_count * sizeof(*processed_table));

        elapsed_ms = KernelLaunch().execute([&]() {
            dim3 blockSize = dim3(32);
            int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
            dim3 gridSize = dim3(bx);
#ifndef opt_kernel
            calculate_obbs_kernel << < gridSize, blockSize >> > (
                object_cu, thrust::raw_pointer_cast(blas->bvh),
                blas->parents, blas->visit_table,
                blas->obbs,
                blas->min_proj, blas->max_proj, blas->minmax_pairs);
#else
            cudaFuncSetCacheConfig(
                calculate_obbs_kernel_shared,
                cudaFuncCachePreferShared);

            const size_t sizeMinProj = sizeof(kr_scalar) * K;
            const size_t sizeMaxProj = sizeof(kr_scalar) * K;
            const size_t sizeMinMaxPair = sizeof(kr_minmax_pair) * K;

            const size_t cacheSize = blockSize.x * (
                sizeMinProj + sizeMaxProj + sizeMinMaxPair);

            calculate_obbs_kernel_shared<<<gridSize, blockSize, cacheSize>>>(
                object_cu, thrust::raw_pointer_cast(blas->bvh),
                blas->parents, blas->visit_table,
                blas->obbs,
                blas->min_proj, blas->max_proj, blas->minmax_pairs);
#endif
        });

        metrics->obb_projection_time += elapsed_ms;
        kr_log("Projection calculation took %fms\n", elapsed_ms);

        elapsed_ms = KernelLaunch().execute([&]() {
            dim3 blockSize = dim3(32);
            int bx = (internal_count + blockSize.x - 1) / blockSize.x;
            dim3 gridSize = dim3(bx);
            calculate_candidate_obb_kernel <<< gridSize, blockSize >>> (
                object_cu,
                blas->obbs, kr_null,
                blas->min_proj, blas->max_proj,
                blas->minmax_pairs);
        });

        metrics->obb_candidates_eval_time += elapsed_ms;
        kr_log("OBB candidates calculation took %fms\n", elapsed_ms);

        cu_error = cudaMemset(blas->visit_table, 0, blas->internal_count * sizeof(*blas->visit_table));
#if 0
        for (int i = 0; i < 50; i++) {
            cu_error = cudaMemset(blas->visit_table, 0, blas->internal_count * sizeof(*blas->visit_table));
            
            elapsed_ms = KernelLaunch().execute([&]() {
                dim3 blockSize = dim3(32);
                int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
                dim3 gridSize = dim3(bx);
                refit_obbs_kernel << < gridSize, blockSize >> > (
                    object_cu, thrust::raw_pointer_cast(blas->bvh),
                    blas->obbs,
                    blas->parents, blas->primitive_counts,
                    processed_table, blas->visit_table,
                    blas->min_proj, blas->max_proj);
            });
            kr_log("OBB refit step[%d] took %fms\n", i, elapsed_ms);
        }
#else
        elapsed_ms = KernelLaunch().execute([&]() {
            dim3 blockSize = dim3(128);
            int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
            dim3 gridSize = dim3(bx);
            refit_obbs_kernel_new << < gridSize, blockSize >> > (
                object_cu, thrust::raw_pointer_cast(blas->bvh),
                blas->obbs, blas->obb_transforms, kr_null,
                blas->parents, blas->primitive_counts,
                kr_null, blas->visit_table,
                blas->min_proj, blas->max_proj);
        });

        metrics->obb_refit_time += elapsed_ms;
        kr_log("OBB refit step new took %fms\n", elapsed_ms);

        for (int i = 0; i < 1; i++) {
            elapsed_ms = KernelLaunch().execute([&]() {
                dim3 blockSize = dim3(512);
                int bx = (internal_count + blockSize.x - 1) / blockSize.x;
                dim3 gridSize = dim3(bx);
                finalize_obbs_kernel << < gridSize, blockSize >> > (
                    object_cu, thrust::raw_pointer_cast(blas->bvh),
                    blas->min_proj, blas->max_proj,
                    blas->obbs, blas->obb_transforms);
                });

            metrics->obb_finalize_time += elapsed_ms;
            kr_log("OBB finalize step new took %fms\n", elapsed_ms);
        }
#endif
    }
    cu_error = cudaDeviceSynchronize();

    //cu_error = cudaMemcpy(h_min_proj, blas->min_proj, K * blas->internal_count * sizeof(*h_min_proj), cudaMemcpyDeviceToHost);
    //cu_error = cudaMemcpy(h_max_proj, blas->max_proj, K * blas->internal_count * sizeof(*h_max_proj), cudaMemcpyDeviceToHost);
    //cu_error = cudaMemcpy(h_minmax_pairs, blas->minmax_pairs, K * blas->internal_count * sizeof(*h_minmax_pairs), cudaMemcpyDeviceToHost);

    cu_error = cudaMemcpy(h_min_proj, blas->min_proj, K * blas->node_count * sizeof(*h_min_proj), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_max_proj, blas->max_proj, K * blas->node_count * sizeof(*h_max_proj), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_minmax_pairs, blas->minmax_pairs, K * blas->node_count * sizeof(*h_minmax_pairs), cudaMemcpyDeviceToHost);

    cu_error = cudaMemcpy(h_obbs, blas->obbs, blas->node_count * sizeof(*h_obbs), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_trasnfroms, blas->obb_transforms, blas->node_count * sizeof(*h_trasnfroms), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_bvh, blas->bvh, blas->node_count * sizeof(*h_bvh), cudaMemcpyDeviceToHost);

    printf("OURS: Fixed Projections\n");
    const auto node_index = 0;
    for (int i = 0; i < 7; i++) {
        printf("OURS: Proj[%d] %f %f\n", i, h_min_proj[node_index * K + i], h_max_proj[node_index * K + i]);
    }
    printf("OURS: Fixed Points\n");
    for (int i = 0; i < 7; i++) {
        printf("OURS: Proj[%d] {%f %f %f} {%f %f %f}\n", i,
            h_minmax_pairs[node_index * K + i].min.x, h_minmax_pairs[node_index * K + i].min.y, h_minmax_pairs[node_index * K + i].min.z,
            h_minmax_pairs[node_index * K + i].max.x, h_minmax_pairs[node_index * K + i].max.y, h_minmax_pairs[node_index * K + i].max.z);
    }

    kr_object* obj = &ads->scene->objects[0];
    kr_obb3 obb_cpu = kr_points_obb(obj->as_mesh.vertices, obj->as_mesh.attr_count, kr_null);

    kr_obb3 obb = obb_cpu;
    printf("Computed OBB CPU:\n");
    printf("Midpoint: %f %f %f\n", obb.mid.x, obb.mid.y, obb.mid.z);
    printf("v0: %f %f %f\n", obb.v0.x, obb.v0.y, obb.v0.z);
    printf("v1: %f %f %f\n", obb.v1.x, obb.v1.y, obb.v1.z);
    printf("v2: %f %f %f\n", obb.v2.x, obb.v2.y, obb.v2.z);
    printf("ext: %f %f %f\n", obb.ext.x, obb.ext.y, obb.ext.z);
    printf("Area: %f\n", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));

    obb = *h_obbs;

    printf("\n\nOURS: OBB Mid %f %f %f\n", obb.mid.x, obb.mid.y, obb.mid.z);
    printf("OURS: OBB Ext %f %f %f\n", obb.ext.x, obb.ext.y, obb.ext.z);
    printf("OURS: OBB A0  %f %f %f\n", obb.v0.x, obb.v0.y, obb.v0.z);
    printf("OURS: OBB A1  %f %f %f\n", obb.v1.x, obb.v1.y, obb.v1.z);
    printf("OURS: OBB A2  %f %f %f\n", obb.v2.x, obb.v2.y, obb.v2.z);
    printf("OURS: OBB Area %f\n", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));

    /*obb = obb_pairs->ub;
    printf("\n\nOURS: OBB Mid %f %f %f\n", obb.mid.x, obb.mid.y, obb.mid.z);
    printf("OURS: OBB Ext %f %f %f\n", obb.ext.x, obb.ext.y, obb.ext.z);
    printf("OURS: OBB A0  %f %f %f\n", obb.v0.x, obb.v0.y, obb.v0.z);
    printf("OURS: OBB A1  %f %f %f\n", obb.v1.x, obb.v1.y, obb.v1.z);
    printf("OURS: OBB A2  %f %f %f\n", obb.v2.x, obb.v2.y, obb.v2.z);
    printf("OURS: OBB Area %f\n", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));
    */

    //kr_bvh_nodes_export(h_bvh, h_bvh, h_trasnfroms, kr_null, 10, "obbs_luda");
    //kr_bvh_nodes_export(h_bvh, h_bvh, kr_null, kr_null, 10, "aabbs_luda");

    kr_cuda_free((void**)&blas->min_proj);
    kr_cuda_free((void**)&blas->max_proj);
    kr_cuda_free((void**)&blas->minmax_pairs);
    //exit(0);
    //system("PAUSE");
    return kr_success;
}

kr_internal kr_error
lbvh_cuda_calculate_codes(kr_ads_lbvh_cuda* ads) {
    kr_scene* scene = ads->scene;

    kr_size instance_count = scene->instance_count;

    /* Calculate the bouunding box of the centroids */
    aabb3 cell = kr_aabb_empty3();
    for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
        kr_object_instance* instance = &scene->instances[instance_index];
        kr_object* object = &scene->objects[instance->object_id];

        switch (object->type) {
        case KR_OBJECT_AABB: {
            cvec3 center = kr_aabb_center3(object->aabb);
            cell = kr_aabb_expand3(cell, center);
            break;
        }
        case KR_OBJECT_MESH: {
            kr_u32 face_count = (kr_u32)object->as_mesh.face_count;
            for (u32 face_index = 0; face_index < face_count; face_index++) {
                uvec4 face = object->as_mesh.faces[face_index];
                vec3  va = object->as_mesh.vertices[face.x];
                vec3  vb = object->as_mesh.vertices[face.y];
                vec3  vc = object->as_mesh.vertices[face.z];
                aabb3 bbox = kr_aabb_empty3();

                bbox = kr_aabb_expand3(bbox, va);
                bbox = kr_aabb_expand3(bbox, vb);
                bbox = kr_aabb_expand3(bbox, vc);
                bbox = kr_aabb_transform4(instance->model.from, bbox);

                cvec3 center = kr_aabb_center3(bbox);

                cell = kr_aabb_expand3(cell, center);
            }
            break;
        } 
        default:
            break;
        }
    }

    ads->centroid_aabb = cell;

    {
      //dim3 blockSize = dim3(1024);
      //int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
      //dim3 gridSize = dim3(bx);
      //mortons_kernel <<< gridSize, blockSize >>> (ads->scene_cu, thrust::raw_pointer_cast(d_codes));
    }

#if 0
    kr_morton* mortons = ads->mortons;
    kr_bvh_node* leaf_nodes = ads->bvh + ads->internal_count;
    for (u32 instance_index = 0, primitive_index = 0; instance_index < instance_count; instance_index++) {
        kr_object_instance* instance = &scene->instances[instance_index];
        kr_object* object = &scene->objects[instance->object_id];

        switch (object->type) {
        case KR_OBJECT_AABB: {
            aabb3 bbox = object->aabb;
            bbox = kr_aabb_transform4(instance->model.from, bbox);

            ads->object_handles[primitive_index] = object;

            leaf_nodes[primitive_index].bounds = bbox;
            leaf_nodes[primitive_index].nPrimitives = 1;
            leaf_nodes[primitive_index].primitivesOffset = primitive_index;

            bbox = kr_aabb_offset3(bbox, kr_vnegate3(cell.min));
            vec3 center = kr_aabb_center3(bbox);
            center = kr_vdiv3s(center, kr_vsub3(cell.max, cell.min));
            kr_morton morton = kr_vmorton3(center);
            mortons[primitive_index] = morton;
            primitive_index++;

            break;
        }
        case KR_OBJECT_MESH: {
            kr_u32 face_count = (kr_u32)object->as_mesh.face_count;
            for (u32 face_index = 0; face_index < face_count; face_index++) {
                uvec4 face = object->as_mesh.faces[face_index];
                vec3  va = object->as_mesh.vertices[face.x];
                vec3  vb = object->as_mesh.vertices[face.y];
                vec3  vc = object->as_mesh.vertices[face.z];
                aabb3 bbox = kr_aabb_create3(va, va);

                bbox = kr_aabb_expand3(bbox, vb);
                bbox = kr_aabb_expand3(bbox, vc);
                bbox = kr_aabb_transform4(instance->model.from, bbox);

                ads->object_handles[primitive_index] = object;

                leaf_nodes[primitive_index].bounds = bbox;
                leaf_nodes[primitive_index].nPrimitives = 1;
                leaf_nodes[primitive_index].primitivesOffset = primitive_index;

                ads->primitives[primitive_index] = primitive_index;

                bbox = kr_aabb_offset3(bbox, kr_vnegate3(cell.min));
                vec3 center = kr_aabb_center3(bbox);
                center = kr_vdiv3s(center, kr_vsub3(cell.max, cell.min));
                kr_morton morton = kr_vmorton3(center);
                mortons[primitive_index] = morton;
                primitive_index++;
            }
            break;
        }
        default:
            break;
        }
    }
#endif
    return kr_success;
}

std::ostream& operator <<(std::ostream& o, DiTO::Vector<float>& u)
{
    return o << '(' << u.x << ',' << u.y << ',' << u.z << ')';
}

kr_inline_host_device
kr_scalar getQualityValue(const vec3& len)
{
    return len.x * len.y + len.x * len.z + len.y * len.z; //half box area
//return len.x * len.y * len.z; //box volume
}

kr_inline_device
void findExtremalPoints_OneDir(vec3& normal, cvec3* vertArr, int nv,
    kr_scalar& minProj, kr_scalar& maxProj, vec3& minVert, vec3& maxVert)
{
    kr_scalar proj = kr_vdot3(vertArr[0], normal);

    // Declare som local variables to avoid aliasing problems
    kr_scalar tMinProj = proj, tMaxProj = proj;
    vec3 tMinVert = vertArr[0], tMaxVert = vertArr[0];

    for (int i = 1; i < nv; i++)
    {
        proj = kr_vdot3(vertArr[i], normal);
        if (proj < tMinProj) { tMinProj = proj; tMinVert = vertArr[i]; }
        if (proj > tMaxProj) { tMaxProj = proj; tMaxVert = vertArr[i]; }
    }

    // Transfer the result to the caller
    minProj = tMinProj;
    maxProj = tMaxProj;
    minVert = tMinVert;
    maxVert = tMaxVert;
}

kr_inline_device
void findExtremalProjs_OneDir(cvec3& normal, cvec3* vertArr, int nv, kr_scalar& minProj, kr_scalar& maxProj)
{
    kr_scalar proj = kr_vdot3(vertArr[0], normal);
    kr_scalar tMinProj = proj, tMaxProj = proj;

    for (int i = 1; i < nv; i++)
    {
        proj = kr_vdot3(vertArr[i], normal);
        tMinProj = min(tMinProj, proj);
        tMaxProj = max(tMaxProj, proj);
    }

    minProj = tMinProj;
    maxProj = tMaxProj;
}

kr_inline_device kr_scalar
findFurthestPointFromInfiniteEdge(vec3& p0, vec3& e0,
    cvec3* vertArr, int nv, vec3& p)
{
    kr_scalar sqDist, maxSqDist;
    int maxIndex = 0;

    maxSqDist = kr_vdistance_to_inf_edge3(vertArr[0], p0, e0);

    for (int i = 1; i < nv; i++)
    {
        sqDist = kr_vdistance_to_inf_edge3(vertArr[i], p0, e0);
        if (sqDist > maxSqDist)
        {
            maxSqDist = sqDist;
            maxIndex = i;
        }
    }
    p = vertArr[maxIndex];
    return maxSqDist;
}

kr_inline_device
void findFurthestPointPair(cvec3* minVert, cvec3* maxVert, int n,
    vec3& p0, vec3& p1)
{
    int indexFurthestPair = 0;
    kr_scalar sqDist, maxSqDist;
    maxSqDist = kr_vdistance3sqr(maxVert[0], minVert[0]);
    //printf("GPU SqDistance[%d] %f Min: {%f %f %f} Max: {%f %f %f}\n", 0, maxSqDist, minVert[0].x, minVert[0].y, minVert[0].z, maxVert[0].x, maxVert[0].y, maxVert[0].z);
    for (int k = 1; k < n; k++)
    {
        sqDist = kr_vdistance3sqr(maxVert[k], minVert[k]);
        //printf("GPU SqDistance[%d] %f Min: {%f %f %f} Max: {%f %f %f}\n", k, sqDist, minVert[k].x, minVert[k].y, minVert[k].z, maxVert[k].x, maxVert[k].y, maxVert[k].z);
        if (sqDist > maxSqDist) { maxSqDist = sqDist; indexFurthestPair = k; }
    }
    p0 = minVert[indexFurthestPair];
    p1 = maxVert[indexFurthestPair];
}

kr_inline_device
void findBestObbAxesFromTriangleNormalAndEdgeVectors(cvec3* vertArr, int nv, vec3& n,
    vec3& e0, vec3& e1, vec3& e2, vec3& b0, vec3& b1, vec3& b2, kr_scalar& bestVal)
{
    vec3 m0, m1, m2;
    vec3 dmax, dmin, dlen;
    kr_scalar quality;

    m0 = kr_vcross3(e0, n);
    m1 = kr_vcross3(e1, n);
    m2 = kr_vcross3(e2, n);

    // The operands are assumed to be orthogonal and unit normals	
    findExtremalProjs_OneDir(n, vertArr, nv, dmin.y, dmax.y);
    dlen.y = dmax.y - dmin.y;

    findExtremalProjs_OneDir(e0, vertArr, nv, dmin.x, dmax.x);
    findExtremalProjs_OneDir(m0, vertArr, nv, dmin.z, dmax.z);
    dlen.x = dmax.x - dmin.x;
    dlen.z = dmax.z - dmin.z;
    quality = getQualityValue(dlen);
    if (quality < bestVal) { bestVal = quality; b0 = e0; b1 = n; b2 = m0; }

    findExtremalProjs_OneDir(e1, vertArr, nv, dmin.x, dmax.x);
    findExtremalProjs_OneDir(m1, vertArr, nv, dmin.z, dmax.z);
    dlen.x = dmax.x - dmin.x;
    dlen.z = dmax.z - dmin.z;
    quality = getQualityValue(dlen);
    if (quality < bestVal) { bestVal = quality; b0 = e1; b1 = n; b2 = m1; }

    findExtremalProjs_OneDir(e2, vertArr, nv, dmin.x, dmax.x);
    findExtremalProjs_OneDir(m2, vertArr, nv, dmin.z, dmax.z);
    dlen.x = dmax.x - dmin.x;
    dlen.z = dmax.z - dmin.z;
    quality = getQualityValue(dlen);
    if (quality < bestVal) { bestVal = quality; b0 = e2; b1 = n; b2 = m2; }

}

kr_inline_device
void findUpperLowerTetraPoints(vec3& n, cvec3* selVertPtr, int np, vec3& p0,
    vec3& p1, vec3& p2, vec3& q0, vec3& q1, int& q0Valid, int& q1Valid)
{
    kr_scalar qMaxProj, qMinProj, triProj;
    kr_scalar eps = 0.000001f;

    q0Valid = q1Valid = 0;

    findExtremalPoints_OneDir(n, selVertPtr, np, qMinProj, qMaxProj, q1, q0);
    triProj = kr_vdot3(p0, n);

    if (qMaxProj - eps > triProj) { q0Valid = 1; }
    if (qMinProj + eps < triProj) { q1Valid = 1; }
}


kr_inline_device
void findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(cvec3* selVertPtr, int np,
    vec3& n, vec3& p0, vec3& p1, vec3& p2, vec3& e0, vec3& e1,
    vec3& e2, vec3& b0, vec3& b1, vec3& b2, kr_scalar& bestVal, kr_obb3& obb)
{
    vec3 q0, q1;     // Top and bottom vertices for lower and upper tetra constructions
    vec3 f0, f1, f2; // Edge vectors towards q0; 
    vec3 g0, g1, g2; // Edge vectors towards q1; 
    vec3 n0, n1, n2; // Unit normals of top tetra tris
    vec3 m0, m1, m2; // Unit normals of bottom tetra tris		

    // Find furthest points above and below the plane of the base triangle for tetra constructions 
    // For each found valid point, search for the best OBB axes based on the 3 arising triangles
    int q0Valid, q1Valid;
    findUpperLowerTetraPoints(n, selVertPtr, np, p0, p1, p2, q0, q1, q0Valid, q1Valid);
    if (q0Valid)
    {
        f0 = kr_vnormalize3(kr_vsub3(q0, p0));
        f1 = kr_vnormalize3(kr_vsub3(q0, p1));
        f2 = kr_vnormalize3(kr_vsub3(q0, p2));
        n0 = kr_vnormalize3(kr_vcross3(f1, e0));
        n1 = kr_vnormalize3(kr_vcross3(f2, e1));
        n2 = kr_vnormalize3(kr_vcross3(f0, e2));
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n0, e0, f1, f0, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n1, e1, f2, f1, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n2, e2, f0, f2, b0, b1, b2, bestVal);
    }
    if (q1Valid)
    {
        g0 = kr_vnormalize3(kr_vsub3(q1, p0));
        g1 = kr_vnormalize3(kr_vsub3(q1, p1));
        g2 = kr_vnormalize3(kr_vsub3(q1, p2));
        m0 = kr_vnormalize3(kr_vcross3(g1, e0));
        m1 = kr_vnormalize3(kr_vcross3(g2, e1));
        m2 = kr_vnormalize3(kr_vcross3(g0, e2));
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m0, e0, g1, g0, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m1, e1, g2, g1, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m2, e2, g0, g2, b0, b1, b2, bestVal);
    }
}

kr_inline_device
int findBestObbAxesFromBaseTriangle(cvec3* minVert, cvec3* maxVert, int ns,
    cvec3* selVertPtr, int np, vec3& n, vec3& p0, vec3& p1, vec3& p2,
    vec3& e0, vec3& e1, vec3& e2, vec3& b0, vec3& b1, vec3& b2, kr_scalar& bestVal, kr_obb3& obb)
{
    kr_scalar sqDist;
    kr_scalar eps = 0.000001f;

    // Find the furthest point pair among the selected min and max point pairs
    findFurthestPointPair(minVert, maxVert, ns, p0, p1);
    
    // Degenerate case 1:
    // If the found furthest points are located very close, return OBB aligned with the initial AABB 
    if (kr_vdistance3sqr(p0, p1) < eps) { return 1; }

    // Compute edge vector of the line segment p0, p1 		
    e0 = kr_vnormalize3(kr_vsub3(p0, p1));

    // Find a third point furthest away from line given by p0, e0 to define the large base triangle
    sqDist = findFurthestPointFromInfiniteEdge(p0, e0, selVertPtr, np, p2);

    // Degenerate case 2:
    // If the third point is located very close to the line, return an OBB aligned with the line 
    if (sqDist < eps) { return 2; }

    // Compute the two remaining edge vectors and the normal vector of the base triangle				
    e1 = kr_vnormalize3(kr_vsub3(p1, p2));
    e2 = kr_vnormalize3(kr_vsub3(p2, p0));
    n = kr_vnormalize3(kr_vcross3(e1, e0));

    // Find best OBB axes based on the base triangle
    findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n, e0, e1, e2, b0, b1, b2, bestVal);

    return 0; // success
}

template <int K = 7>
__global__ void
dito_obb_compute(
    const kr_vec3* points, int N,
    kr_scalar* minProj, kr_scalar* maxProj,
    kr_obb3* obb
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= N)
        return;

    vec3 bMin, bMax, bLen; // The dimensions of the oriented box

    cvec3 b0 = obb->v0;
    cvec3 b1 = obb->v1;
    cvec3 b2 = obb->v2;

    cvec3 point = points[point_index];

    kr_scalar proj;

    // We use some local variables to avoid aliasing problems
    kr_scalar* tMinProj = minProj;
    kr_scalar* tMaxProj = maxProj;

    proj = kr_vdot3(point, b0);
    atomicMin(tMinProj + 0, proj);
    atomicMax(tMaxProj + 0, proj);

    proj = kr_vdot3(point, b1);
    atomicMin(tMinProj + 1, proj);
    atomicMax(tMaxProj + 1, proj);

    proj = kr_vdot3(point, b2);
    atomicMin(tMinProj + 2, proj);
    atomicMax(tMaxProj + 2, proj);

}

template <int K = 7>
__global__ void
dito_obb_candidate(
    const kr_vec3* points, int N,
    kr_scalar* minProj, kr_scalar* maxProj,
    kr_vec3* gminVert, kr_vec3* gmaxVert,
    kr_i32* argMinVert, kr_i32* argMaxVert,
    kr_obb3* obb
 ) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= K)
        return;

    gminVert[point_index] = points[argMinVert[point_index]];
    gmaxVert[point_index] = points[argMaxVert[point_index]];

    if (point_index >= 1)
        return;

    vec3 minVert[K], maxVert[K];

    const kr_vec3* selVert = minVert;
    for (int i = 0; i < K; i++) {
        minVert[i] = points[argMinVert[i]];
        maxVert[i] = points[argMaxVert[i]];
    }

    int np = (N > 2 * K) ? 2 * K : N;
    const kr_vec3* selVertPtr = (N > 2 * K) ? minVert : points;

    vec3 p0, p1, p2; // Vertices of the large base triangle
    vec3 e0, e1, e2; // Edge vectors of the large base triangle
    vec3 n;
    vec3 bMin, bMax, bLen; // The dimensions of the oriented box
    kr_obb3 obb_candidate;

    vec3 alMid = { (minProj[0] + maxProj[0]) * 0.5f, (minProj[1] + maxProj[1]) * 0.5f, (minProj[2] + maxProj[2]) * 0.5f };
    vec3 alLen = { maxProj[0] - minProj[0], maxProj[1] - minProj[1], maxProj[2] - minProj[2] };
    kr_scalar alVal = getQualityValue(alLen);

    // Initialize the best found orientation so far to be the standard base
    kr_scalar bestVal = alVal;
    vec3 b0 = { 1, 0, 0 };
    vec3 b1 = { 0, 1, 0 };
    vec3 b2 = { 0, 0, 1 };

    int baseTriangleConstr = findBestObbAxesFromBaseTriangle(minVert, maxVert, K, selVertPtr, np, n, p0, p1, p2, e0, e1, e2, b0, b1, b2, bestVal, obb_candidate);
    /* Handle degenerate case */


    if (point_index == 0) {

        printf("Base Trinagle %d %f:\n", baseTriangleConstr, bestVal);
        printf("Normal: {%f %f %f}\n", n.x, n.y, n.z);
        printf("P0	  : {%f %f %f}\n", p0.x, p0.y, p0.z);
        printf("P1    : {%f %f %f}\n", p1.x, p1.y, p1.z);
        printf("P2    : {%f %f %f}\n", p2.x, p2.y, p2.z);
        printf("B0	  : {%f %f %f}\n", b0.x, b0.y, b0.z);
        printf("B1    : {%f %f %f}\n", b1.x, b1.y, b1.z);
        printf("B2    : {%f %f %f}\n", b2.x, b2.y, b2.z);
        printf("E0	  : {%f %f %f}\n", e0.x, e0.y, e0.z);
        printf("E1    : {%f %f %f}\n", e1.x, e1.y, e1.z);
        printf("E2    : {%f %f %f}\n", e2.x, e2.y, e2.z);

        obb->v0 = b0;
        obb->v1 = b1;
        obb->v2 = b2;
        obb->ext = { 1, 1, 1 };
        obb->mid = { 0, 0, 0 };
    }
    // Find improved OBB axes based on constructed di-tetrahedral shape raised from base triangle
    findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(selVertPtr, np, n, p0, p1, p2, e0, e1, e2, b0, b1, b2, bestVal, obb_candidate);

    if (point_index == 0) {

        printf("Tetras %d %f:\n", baseTriangleConstr, bestVal);
        printf("Normal: {%f %f %f}\n", n.x, n.y, n.z);
        printf("P0	  : {%f %f %f}\n", p0.x, p0.y, p0.z);
        printf("P1    : {%f %f %f}\n", p1.x, p1.y, p1.z);
        printf("P2    : {%f %f %f}\n", p2.x, p2.y, p2.z);
        printf("B0	  : {%f %f %f}\n", b0.x, b0.y, b0.z);
        printf("B1    : {%f %f %f}\n", b1.x, b1.y, b1.z);
        printf("B2    : {%f %f %f}\n", b2.x, b2.y, b2.z);
        printf("E0	  : {%f %f %f}\n", e0.x, e0.y, e0.z);
        printf("E1    : {%f %f %f}\n", e1.x, e1.y, e1.z);
        printf("E2    : {%f %f %f}\n", e2.x, e2.y, e2.z);

        obb->v0 = b0;
        obb->v1 = b1;
        obb->v2 = b2;
        obb->ext = { 1, 1, 1 };
        obb->mid = { 0, 0, 0 };
    }

    //computeObbDimensions(points, N, b0, b1, b2, bMin, bMax);

    bLen = kr_vsub3(bMax, bMin);
    bestVal = getQualityValue(bLen);
}

template <
    int THREAD_BLOCK_SIZE = 256
>
kr_inline_device
kr_scalar dito_block_min(kr_scalar* cache, kr_scalar value) {

    // Perform parallel reduction within the block.
    // TODO: Replace with shuffle down sync operations for the 32 block
    cache[threadIdx.x] = value;
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 1]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 2]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 4]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 8]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 128]);

    return cache[threadIdx.x];
}

template <
    int THREAD_BLOCK_SIZE = 256
>
kr_inline_device
kr_scalar dito_block_max(kr_scalar* cache, kr_scalar value) {

    // Perform parallel reduction within the block.
    // TODO: Replace with shuffle down sync operations for the 32 block
    cache[threadIdx.x] = value;
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 1]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 2]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 4]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 8]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 128]);

    return cache[threadIdx.x];
}

template <
    int K = 7,
    int THREAD_BLOCK_SIZE = 256,
    int POINT_BLOCK_SIZE = 4
>
__global__ void
dito_minmax_vert(
    cvec3* points, int N,
    const kr_scalar* minProj, const kr_scalar* maxProj,
    kr_i32* argMinVert, kr_i32* argMaxVert
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= N)
        return;

    cvec3 point = points[point_index];
    kr_scalar proj;

    // Slab 0: dir {1, 0, 0}
    proj = point.x;
    if (proj == minProj[0]) { atomicExch(&argMinVert[0], point_index); }
    if (proj == maxProj[0]) { atomicExch(&argMaxVert[0], point_index); }
    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    if (proj == minProj[1]) { atomicExch(&argMinVert[1], point_index); }
    if (proj == maxProj[1]) { atomicExch(&argMaxVert[1], point_index); }
    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    if (proj == minProj[2]) { atomicExch(&argMinVert[2], point_index); }
    if (proj == maxProj[2]) { atomicExch(&argMaxVert[2], point_index); }
    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    if (proj == minProj[3]) { atomicExch(&argMinVert[3], point_index); }
    if (proj == maxProj[3]) { atomicExch(&argMaxVert[3], point_index); }
    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    if (proj == minProj[4]) { atomicExch(&argMinVert[4], point_index); }
    if (proj == maxProj[4]) { atomicExch(&argMaxVert[4], point_index); }
    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    if (proj == minProj[5]) { atomicExch(&argMinVert[5], point_index); }
    if (proj == maxProj[5]) { atomicExch(&argMaxVert[5], point_index); }
    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    if (proj == minProj[6]) { atomicExch(&argMinVert[6], point_index); }
    if (proj == maxProj[6]) { atomicExch(&argMaxVert[6], point_index); }
}

template <
    int K = 7, 
    int THREAD_BLOCK_SIZE = 256, 
    int POINT_BLOCK_SIZE = 4
>
__global__ void
dito_minmax_proj(
    const kr_vec3* points, int N,
    kr_scalar* minProj, kr_scalar* maxProj
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= N)
        return;

    cvec3 point = points[point_index];
    kr_scalar proj;

    __shared__ kr_scalar proj_cache[THREAD_BLOCK_SIZE];

    // Slab 0: dir {1, 0, 0}
    proj = point.x;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[0], proj_cache[threadIdx.x]);
    }
    proj = point.x;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[0], proj_cache[threadIdx.x]);
    }

    __syncthreads();

    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[1], proj_cache[threadIdx.x]);
    }
    proj = point.y;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[1], proj_cache[threadIdx.x]);
    }

    __syncthreads();

    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[2], proj_cache[threadIdx.x]);
    }
    proj = point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[2], proj_cache[threadIdx.x]);
    }

    __syncthreads();

    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[3], proj_cache[threadIdx.x]);
    }
    proj = point.x + point.y + point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[3], proj_cache[threadIdx.x]);
    }

    __syncthreads();

    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[4], proj_cache[threadIdx.x]);
    }
    proj = point.x + point.y - point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[4], proj_cache[threadIdx.x]);
    }

    __syncthreads();

    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[5], proj_cache[threadIdx.x]);
    }
    proj = point.x - point.y + point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[5], proj_cache[threadIdx.x]);
    }

    __syncthreads();

    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[6], proj_cache[threadIdx.x]);
    }
    proj = point.x - point.y - point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[6], proj_cache[threadIdx.x]);
    }

#if 0
    // Perform parallel reduction within the block.
    bound[threadIdx.x] = proj;
    bound[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 1]);
    bound[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 2]);
    bound[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 4]);
    bound[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 8]);
    bound[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) proj_cache[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) proj_cache[threadIdx.x] = kr_min(proj_cache[threadIdx.x], proj_cache[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) proj_cache[threadIdx.x] = kr_min(bound[threadIdx.x], proj_cache[threadIdx.x ^ 128]);

    // Update global bounding box.
    if (threadIdx.x == 0) {
        atomicMin(&scene_bbox->min.x, bound[threadIdx.x].x);
    }
#endif

#if 0
    // We use some local variables to avoid aliasing problems
    kr_scalar* tMinProj = minProj;
    kr_scalar* tMaxProj = maxProj;

    //kr_vec3*   tMinVert[K], tMaxVert[K];

    // Slab 0: dir {1, 0, 0}
    proj = point.x;
    atomicMin(tMinProj + 0, proj);
    atomicMax(tMaxProj + 0, proj);
    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    atomicMin(tMinProj + 1, proj);
    atomicMax(tMaxProj + 1, proj);
    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    atomicMin(tMinProj + 2, proj);
    atomicMax(tMaxProj + 2, proj);
    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    atomicMin(tMinProj + 3, proj);
    atomicMax(tMaxProj + 3, proj);
    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    atomicMin(tMinProj + 4, proj);
    atomicMax(tMaxProj + 4, proj);
    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    atomicMin(tMinProj + 5, proj);
    atomicMax(tMaxProj + 5, proj);
    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    atomicMin(tMinProj + 6, proj);
    atomicMax(tMaxProj + 6, proj);

    __threadfence();

    // Slab 0: dir {1, 0, 0}
    proj = point.x;
    if (proj == tMinProj[0]) { atomicExch(&argMinVert[0], point_index); }
    if (proj == tMaxProj[0]) { atomicExch(&argMaxVert[0], point_index); }
    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    if (proj == tMinProj[1]) { atomicExch(&argMinVert[1], point_index); }
    if (proj == tMaxProj[1]) { atomicExch(&argMaxVert[1], point_index); }
    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    if (proj == tMinProj[2]) { atomicExch(&argMinVert[2], point_index); }
    if (proj == tMaxProj[2]) { atomicExch(&argMaxVert[2], point_index); }
    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    if (proj == tMinProj[3]) { atomicExch(&argMinVert[3], point_index); }
    if (proj == tMaxProj[3]) { atomicExch(&argMaxVert[3], point_index); }
    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    if (proj == tMinProj[4]) { atomicExch(&argMinVert[4], point_index); }
    if (proj == tMaxProj[4]) { atomicExch(&argMaxVert[4], point_index); }
    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    if (proj == tMinProj[5]) { atomicExch(&argMinVert[5], point_index); }
    if (proj == tMaxProj[5]) { atomicExch(&argMaxVert[5], point_index); }
    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    if (proj == tMinProj[6]) { atomicExch(&argMinVert[6], point_index); }
    if (proj == tMaxProj[6]) { atomicExch(&argMaxVert[6], point_index); }
#endif
}

kr_internal kr_error
lbvh_cuda_blas_dito(kr_ads_lbvh_cuda* ads, kr_object* object) {
    cudaError_t cu_error;

    int smemSize, numProcs;
    cudaDeviceGetAttribute(&smemSize,
        cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&numProcs,
        cudaDevAttrMultiProcessorCount, 0);

    kr_log("CUDA Max Shared Memory Per Block %d\n", smemSize);
    kr_log("CUDA Multi Processor Count %d\n", numProcs);

    kr_vec3* points = object->as_mesh.vertices;
    kr_size point_count = object->as_mesh.attr_count;

    auto start_cpu = std::chrono::steady_clock::now();
    kr_obb3 obb_cpu = kr_points_obb(points, point_count, nullptr);
    auto end_cpu = std::chrono::steady_clock::now();

    printf("Elapsed time in milliseconds: %dms\n", 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count());

    kr_obb3 obb_gpu = kr_cuda_points_obb(points, point_count, nullptr);
    {
        kr_obb3 obb = obb_cpu;

        printf("Computed OBB CPU:\n");
        printf("\tMidpoint: %f %f %f", obb.mid.x, obb.mid.y, obb.mid.z);
        printf("\n\tv0: %f %f %f", obb.v0.x, obb.v0.y, obb.v0.z);
        printf("\n\tv1: %f %f %f", obb.v1.x, obb.v1.y, obb.v1.z);
        printf("\n\tv2: %f %f %f", obb.v2.x, obb.v2.y, obb.v2.z);
        printf("\n\text: %f %f %f", obb.ext.x, obb.ext.y, obb.ext.z);
        printf("\n\tArea: %f \n", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));

        obb = obb_gpu;
        printf("Computed OBB GPU:\n");
        printf("\tMidpoint: %f %f %f", obb.mid.x, obb.mid.y, obb.mid.z);
        printf("\n\tv0: %f %f %f", obb.v0.x, obb.v0.y, obb.v0.z);
        printf("\n\tv1: %f %f %f", obb.v1.x, obb.v1.y, obb.v1.z);
        printf("\n\tv2: %f %f %f", obb.v2.x, obb.v2.y, obb.v2.z);
        printf("\n\text: %f %f %f", obb.ext.x, obb.ext.y, obb.ext.z);
        printf("\n\tArea: %f ", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));
    }

    exit(0);

    printf("Point Count %d\n", point_count);

#define KR_DITO_EXTERNAL_POINT_COUNT 7

    KR_ALLOC_DECLARE(kr_vec3, h_points, point_count);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_vec3, d_points, point_count);

    KR_ALLOC_DECLARE(kr_scalar, h_proj_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_scalar, d_proj_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);

    KR_ALLOC_DECLARE(kr_vec3, h_final_vert_minmax, 2);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_vec3, d_final_vert_minmax, 2);

    KR_ALLOC_DECLARE(kr_vec3, h_vert_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_vec3, d_vert_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);

    KR_ALLOC_DECLARE(kr_i32, h_vert_argminmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_i32, d_vert_argminmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);

    KR_ALLOC_DECLARE(kr_obb3, h_obb, sizeof(*h_obb));
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_obb3, d_obb, sizeof(*h_obb));

    memcpy(h_points, points, point_count * sizeof(*h_points));
    cu_error = cudaMemcpy(thrust::raw_pointer_cast(d_points), h_points, point_count * sizeof(*h_points), cudaMemcpyHostToDevice);
    
    thrust::fill(d_vert_argminmax, d_vert_argminmax + 2 * KR_DITO_EXTERNAL_POINT_COUNT, -1);
    thrust::fill(d_final_vert_minmax + 0, d_final_vert_minmax + 1, kr_vec3{ FLT_MAX, FLT_MAX, FLT_MAX });
    thrust::fill(d_final_vert_minmax + 1, d_final_vert_minmax + 2, kr_vec3{ -FLT_MAX, -FLT_MAX, -FLT_MAX });
    thrust::fill(d_proj_minmax + 0, d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT, FLT_MAX);
    thrust::fill(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT, d_proj_minmax + 2 * KR_DITO_EXTERNAL_POINT_COUNT, -FLT_MAX);

#if 1
    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

#define KR_DITO_THREAD_BLOCK_SIZE 256
#define KR_DITO_POINT_BLOCK_SIZE  4
    dim3 blockSize = dim3(KR_DITO_THREAD_BLOCK_SIZE);
    int bx = (point_count + blockSize.x - 1) / blockSize.x;
    dim3 gridSize = dim3(bx);
    dito_minmax_proj
        <KR_DITO_EXTERNAL_POINT_COUNT, KR_DITO_THREAD_BLOCK_SIZE, KR_DITO_POINT_BLOCK_SIZE>
        <<<gridSize, blockSize>>> (
            thrust::raw_pointer_cast(d_points), point_count,
            thrust::raw_pointer_cast(d_proj_minmax), thrust::raw_pointer_cast(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT)
        );

    dito_minmax_vert
        <KR_DITO_EXTERNAL_POINT_COUNT, KR_DITO_THREAD_BLOCK_SIZE, KR_DITO_POINT_BLOCK_SIZE>
        <<<gridSize, blockSize>>> (
            thrust::raw_pointer_cast(d_points), point_count,
            thrust::raw_pointer_cast(d_proj_minmax), thrust::raw_pointer_cast(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT),
            thrust::raw_pointer_cast(d_vert_argminmax), thrust::raw_pointer_cast(d_vert_argminmax + KR_DITO_EXTERNAL_POINT_COUNT)
            );

    dito_obb_candidate <<<gridSize, blockSize >>> (
        thrust::raw_pointer_cast(d_points), point_count,
        thrust::raw_pointer_cast(d_proj_minmax), thrust::raw_pointer_cast(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT),
        thrust::raw_pointer_cast(d_vert_minmax), thrust::raw_pointer_cast(d_vert_minmax + KR_DITO_EXTERNAL_POINT_COUNT),
        thrust::raw_pointer_cast(d_vert_argminmax), thrust::raw_pointer_cast(d_vert_argminmax + KR_DITO_EXTERNAL_POINT_COUNT),
        thrust::raw_pointer_cast(d_obb)
    );

    dito_obb_compute << <gridSize, blockSize >> > (
        thrust::raw_pointer_cast(d_points), point_count,
        (kr_scalar*)thrust::raw_pointer_cast(d_final_vert_minmax), (kr_scalar*)thrust::raw_pointer_cast(d_final_vert_minmax + 1),
        thrust::raw_pointer_cast(d_obb)
    );

    cu_error = cudaEventRecord(end);
    cu_error = cudaEventSynchronize(end);
    cu_error = cudaDeviceSynchronize();

    float_t ms = 0.f;
    cu_error = cudaEventElapsedTime(&ms, start, end);
    kr_log("CUDA DiTO took %fms\n", ms);
#endif

    cu_error = cudaMemcpy(h_proj_minmax, thrust::raw_pointer_cast(d_proj_minmax), 2 * KR_DITO_EXTERNAL_POINT_COUNT * sizeof(*h_proj_minmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_vert_argminmax, thrust::raw_pointer_cast(d_vert_argminmax), 2 * KR_DITO_EXTERNAL_POINT_COUNT * sizeof(*h_vert_argminmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_vert_minmax, thrust::raw_pointer_cast(d_vert_minmax), 2 * KR_DITO_EXTERNAL_POINT_COUNT * sizeof(*h_vert_minmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_final_vert_minmax, thrust::raw_pointer_cast(d_final_vert_minmax), 2 * sizeof(*h_final_vert_minmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_obb, thrust::raw_pointer_cast(d_obb), sizeof(*h_obb), cudaMemcpyDeviceToHost);

    printf("GPU values:\n");
    for (int i = 0; i < 7; i++) {
        const auto& minVert = h_vert_minmax[i + 0];
        const auto& maxVert = h_vert_minmax[i + KR_DITO_EXTERNAL_POINT_COUNT];
        printf("Proj[%d] Min: %f Max: %f\n", i, h_proj_minmax[i + 0], h_proj_minmax[i + KR_DITO_EXTERNAL_POINT_COUNT]);
        printf("Vert[%d] Min: %d Max: %d\n", i, h_vert_argminmax[i + 0], h_vert_argminmax[i + KR_DITO_EXTERNAL_POINT_COUNT]);
        printf("Vert[%d] Min: {%f %f %f} Max: {%f %f %f}\n", i, minVert.x, minVert.y, minVert.z, maxVert.x, maxVert.y, maxVert.z);
    }

    /*printf("OBB {%f %f %f}\n\t{%f %f %f}\n\t{%f %f %f}",
        h_obb->v0.x, h_obb->v0.y, h_obb->v0.z,
        h_obb->v1.x, h_obb->v1.y, h_obb->v1.z,
        h_obb->v2.x, h_obb->v2.y, h_obb->v2.z
    );*/

    cu_error = cudaMemcpy(h_points, thrust::raw_pointer_cast(d_points), point_count * sizeof(*h_points), cudaMemcpyDeviceToHost);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    
    DiTO::OBB<float> obb;
    DiTO::DiTO_14<float>((DiTO::Vector<float>*) points, point_count, obb);

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    cvec3 bMin = *(h_final_vert_minmax + 0);
    cvec3 bMax = *(h_final_vert_minmax + 1);
    cvec3 bLen = kr_vsub3(bMax, bMin);
    kr_scalar bestVal = getQualityValue(bLen);
    printf("Fianl Vert Min : {%f %f %f} Max: {%f %f %f}\n", bMin.x, bMin.y, bMin.z, bMax.x, bMax.y, bMax.z);
    printf("Fianl Blen     : {%f %f %f} : %f\n", bLen.x, bLen.y, bLen.z, bestVal);

    /*std::cout << "Computed OBB:\n";
    std::cout << "\tMidpoint:" << obb.mid;
    std::cout << "\n\tv0: " << obb.v0;
    std::cout << "\n\tv1: " << obb.v1;
    std::cout << "\n\tv2: " << obb.v2;
    std::cout << "\n\text: " << obb.ext;
    std::cout << "\n\tArea: " << 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z);
    */
    return kr_success;
}

__global__ void
persistent_prepare_kernel(
    const kr_object_cu* object, 
    kr_bvh_node_packed* packed_nodes,
    const kr_bvh_node* nodes) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 primitive_count = (u32)object->as_mesh.face_count;
    const u32 internal_count = (u32)(primitive_count - 1);
    const u32 node_count = primitive_count + internal_count;
    if (node_index >= node_count)
        return;

    const kr_bvh_node* node = &nodes[node_index];
    kr_bvh_node_packed* packed_node = &packed_nodes[node_index];

    if (node->nPrimitives == 0) {
        const kr_bvh_node* l = &nodes[node->left];
        const kr_bvh_node* r = &nodes[node->right];

        packed_node->lbounds = l->bounds;
        packed_node->rbounds = r->bounds;
        packed_node->left = (l->nPrimitives > 0) ? ~((i32)node->left) : (i32)node->left;
        packed_node->right = (r->nPrimitives > 0) ? ~((i32)node->right) : (i32)node->right;
    }
    else {
        packed_node->lbounds = node->bounds;
        packed_node->left = (i32)node->primitivesOffset;
        packed_node->right = (i32)(node->primitivesOffset + node->nPrimitives);
    }

    packed_node->metadata.z = 0;
}

kr_internal kr_error
lbvh_cuda_blas_prepare_persistent(kr_ads_lbvh_cuda* ads, kr_ads_blas_cuda* blas, kr_object* object) {
    kr_bvh_node* bvh = (kr_null != blas->collapsed_bvh) ? blas->collapsed_bvh : blas->bvh;

    if (!ads->use_obbs) {
        kr_cuda_bvh_persistent(
            bvh,
            blas->bvh_packed,
            blas->node_count
        );
    }
    else {
        blas->transformations = (kr_bvh_transformation_pair*)kr_cuda_allocate(blas->internal_count * sizeof(*blas->transformations));

        kr_cuda_obvh_persistent(
            bvh,
            blas->obb_transforms,
            blas->bvh_packed,
            blas->transformations,
            blas->node_count
        );

        kr_cuda_soa_obvh(
            blas->bvh_packed, blas->transformations,
            blas->soa_obvh_packed,
            blas->internal_count, blas->node_count);

    }

    return kr_success;
}

kr_internal kr_error
lbvh_cuda_blas_commit(kr_ads_lbvh_cuda* ads, kr_ads_blas_cuda* blas, kr_object* object, kr_ads_build_metrics * metrics) {
    kr_vec3* points = object->as_mesh.vertices;
    kr_size point_count = object->as_mesh.attr_count;

    aabb3 scene_bbox = kr_aabb_empty3();
    aabb3 scene_bbox_centroid = kr_aabb_empty3();
    kr_size primitive_count = 0;
    cudaError_t cu_error;
    kr_scalar elapsed_ms = 0.0f;

    kr_object_cu* h_object_cu = (kr_object_cu*)kr_allocate(sizeof(*h_object_cu));
    kr_object_cu* d_object_cu = (kr_object_cu*)kr_cuda_allocate(sizeof(*d_object_cu));
    h_object_cu->type = object->type;

    switch (object->type) {
    case KR_OBJECT_AABB:
        primitive_count = 1;
        break;
    case KR_OBJECT_MESH: {
        kr_u32 face_count = (kr_u32)object->as_mesh.face_count;
        kr_u32 attr_count = (kr_u32)object->as_mesh.attr_count;
        primitive_count = (kr_u32)object->as_mesh.face_count;

        for (u32 face_index = 0; face_index < face_count; face_index++) {
            uvec4 face = object->as_mesh.faces[face_index];
            vec3  va = object->as_mesh.vertices[face.x];
            vec3  vb = object->as_mesh.vertices[face.y];
            vec3  vc = object->as_mesh.vertices[face.z];
            aabb3 bbox = kr_aabb_empty3();

            bbox = kr_aabb_expand3(bbox, va);
            bbox = kr_aabb_expand3(bbox, vb);
            bbox = kr_aabb_expand3(bbox, vc);

            cvec3 center = kr_aabb_center3(bbox);

            scene_bbox = kr_aabb_expand(scene_bbox, bbox);
            scene_bbox_centroid = kr_aabb_expand3(scene_bbox_centroid, center);
        }

        h_object_cu->as_mesh.vertices = (vec3*)kr_cuda_allocate(object->as_mesh.attr_count * sizeof(*h_object_cu->as_mesh.vertices));
        h_object_cu->as_mesh.attr_count = object->as_mesh.attr_count;
        h_object_cu->as_mesh.faces = (uvec4*)kr_cuda_allocate(object->as_mesh.face_count * sizeof(*h_object_cu->as_mesh.faces));
        h_object_cu->as_mesh.face_count = object->as_mesh.face_count;

        cudaMemcpy(h_object_cu->as_mesh.vertices, object->as_mesh.vertices, attr_count * sizeof(*object->as_mesh.vertices), cudaMemcpyHostToDevice);
        cudaMemcpy(h_object_cu->as_mesh.faces, object->as_mesh.faces, face_count * sizeof(*object->as_mesh.faces), cudaMemcpyHostToDevice);
       
        break;
    } 
    default:
        break;
    }

    cudaMemcpy(d_object_cu, h_object_cu, sizeof(*h_object_cu), cudaMemcpyHostToDevice);

    ads->aabb = scene_bbox;

    blas->node_count = 2 * (kr_u32)primitive_count - 1;
    blas->internal_count = (kr_u32)primitive_count - 1;
    blas->primitive_count = (kr_u32)primitive_count;
    blas->leaf_count = (kr_u32)primitive_count;

    //blas->bvh = (kr_bvh_node*)kr_cuda_allocate_managed(blas->node_count * sizeof(*blas->bvh));
    blas->primitive_counts = (u32*)kr_cuda_allocate(blas->node_count * sizeof(*blas->primitive_counts));
    blas->obb_transforms = (kr_mat4*)kr_cuda_allocate(blas->node_count * sizeof(*blas->obb_transforms));
    blas->obbs = (kr_obb3*)kr_cuda_allocate(blas->node_count * sizeof(*blas->obbs));

    blas->mortons = (kr_morton*)kr_cuda_allocate(blas->primitive_count * sizeof(*blas->mortons));
    blas->primitives = (kr_u32*)kr_cuda_allocate(blas->primitive_count * sizeof(*blas->primitives));
    blas->costs = (kr_scalar*)kr_cuda_allocate(blas->node_count * sizeof(*blas->costs));
    blas->parents = (u32*)kr_cuda_allocate(blas->node_count * sizeof(*blas->parents));
    blas->visit_table = (b32*)kr_cuda_allocate(blas->internal_count * sizeof(*blas->visit_table));
    blas->collapse_table = (b32*)kr_cuda_allocate(blas->node_count * sizeof(*blas->collapse_table));

    KR_CUDA_ALLOC_THRUST_DECLARE(u32, d_ray_counter, 1);

    KR_ALLOC_DECLARE(aabb3, h_bbox_scratch, 2);
    KR_CUDA_ALLOC_THRUST_DECLARE(aabb3, d_bbox_scratch, 2);

    KR_ALLOC_DECLARE(kr_morton, h_mortons, primitive_count);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_morton, d_mortons, primitive_count);

    KR_ALLOC_DECLARE(kr_scalar, h_costs, blas->node_count);

    KR_ALLOC_DECLARE(kr_bvh_node, h_bvh, blas->node_count);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_bvh_node, d_bvh, blas->node_count);

    thrust::device_ptr<kr_bvh_node> d_collapsed_bvh = kr_null;
    kr_bvh_node* h_collapsed_bvh = kr_null;
    if (kr_false != ads->collapse) {
        d_collapsed_bvh = thrust::device_ptr<kr_bvh_node>((kr_bvh_node*)kr_cuda_allocate(blas->node_count * sizeof(kr_bvh_node)));
        h_collapsed_bvh = (kr_bvh_node*)kr_allocate(blas->node_count * sizeof(kr_bvh_node));
    }

    KR_ALLOC_DECLARE(kr_bvh_node_packed, h_bvh_packed, blas->node_count);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_bvh_node_packed, d_bvh_packed, blas->node_count);

    thrust::fill(d_bbox_scratch, d_bbox_scratch + 2, kr_aabb_empty3());
    
    kr_cuda_bbox_calculate(
        h_object_cu->as_mesh.faces,
        h_object_cu->as_mesh.vertices,
        h_object_cu->as_mesh.face_count,
        thrust::raw_pointer_cast(d_bbox_scratch)
    );
    kr_cuda_centroid_bbox_calculate(
        h_object_cu->as_mesh.faces,
        h_object_cu->as_mesh.vertices,
        h_object_cu->as_mesh.face_count,
        thrust::raw_pointer_cast(d_bbox_scratch) + 1
    );

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(256);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        mortons_kernel << < gridSize, blockSize >> > (d_object_cu, thrust::raw_pointer_cast(d_mortons), thrust::raw_pointer_cast(d_bvh) + blas->internal_count, thrust::raw_pointer_cast(d_bbox_scratch));
    });
    kr_log("Mortons calculation took %fms\n", elapsed_ms);

    elapsed_ms = KernelLaunch().execute([&]() {
        thrust::stable_sort_by_key(d_mortons, d_mortons + primitive_count, d_bvh + blas->internal_count);
    });
    kr_log("Sorting took %fms\n", elapsed_ms);

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(256);
        int bx = ((primitive_count - 1) + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        radix_tree_build_kernel <<< gridSize, blockSize >>> (d_object_cu, thrust::raw_pointer_cast(d_bvh), thrust::raw_pointer_cast(d_mortons), blas->parents);
    });
    kr_log("Radix tree building took %fms\n", elapsed_ms);
    
    cu_error = cudaMemset(blas->visit_table, 0, blas->internal_count * sizeof(*blas->visit_table));

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(512);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        calculate_aabbs_kernel <<< gridSize, blockSize >>> (d_object_cu, thrust::raw_pointer_cast(d_bvh), blas->parents, blas->primitive_counts, blas->costs, blas->visit_table);
    });
    kr_log("Metrics calculation took %fms\n", elapsed_ms);

    blas->d_object_cu = d_object_cu;
    blas->h_object_cu = h_object_cu;
    blas->bvh = thrust::raw_pointer_cast(d_bvh);
    blas->collapsed_bvh = thrust::raw_pointer_cast(d_collapsed_bvh);
    blas->bvh_packed = thrust::raw_pointer_cast(d_bvh_packed);
    blas->ray_counter = thrust::raw_pointer_cast(d_ray_counter);

    cu_error = cudaMemcpy(h_bbox_scratch, thrust::raw_pointer_cast(d_bbox_scratch), 2 * sizeof(*h_bbox_scratch), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_costs, blas->costs, blas->node_count * sizeof(*h_costs), cudaMemcpyDeviceToHost);

    kr_aabb3 a1 = *h_bbox_scratch;
    kr_aabb3 a2 = *(h_bbox_scratch + 1);
    kr_f32 root_area = kr_aabb_surface_area3(scene_bbox);
    kr_log("LBVH Scene      AABB [%f %f %f] x [%f %f %f]\n", scene_bbox.min.x, scene_bbox.min.y, scene_bbox.min.z, scene_bbox.max.x, scene_bbox.max.y, scene_bbox.max.z);
    kr_log("LBVH Primitive  AABB [%f %f %f] x [%f %f %f]\n", a1.min.x, a1.min.y, a1.min.z, a1.max.x, a1.max.y, a1.max.z);
    kr_log("LBVH Centroid   AABB [%f %f %f] x [%f %f %f]\n", a2.min.x, a2.min.y, a2.min.z, a2.max.x, a2.max.y, a2.max.z);
    kr_log("LBVH CUDA SAH Cost %f\n", h_costs[0] / root_area);
    
    KR_ALLOC_DECLARE(kr_obb3, h_obbs, blas->node_count);
    cudaMemcpy(h_bvh, blas->bvh, blas->node_count * sizeof(kr_bvh_node), cudaMemcpyDeviceToHost);
    kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, nullptr, ads->scene->benchmark_file);

    if (ads->use_obbs)
    {
        lbvh_cuda_calculate_obbs(ads, &metrics->obb_metrics);
        cudaMemcpy(h_obbs, blas->obbs, blas->node_count * sizeof(kr_obb3), cudaMemcpyDeviceToHost);
        kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, h_obbs, ads->scene->benchmark_file);
    }

    if (ads->collapse)
    {
        lbvh_cuda_collapse(ads, blas, metrics);
        cudaMemcpy(h_bvh, blas->collapsed_bvh, blas->node_count * sizeof(kr_bvh_node), cudaMemcpyDeviceToHost);
        kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, nullptr, ads->scene->benchmark_file);

        if (ads->use_obbs)
        {
            kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, h_obbs, ads->scene->benchmark_file);
        }
    }

    kr_free((void**)&h_obbs);
    cu_error = cudaMemcpy(h_costs, blas->costs, blas->node_count * sizeof(*h_costs), cudaMemcpyDeviceToHost);

    kr_log("LBVH CUDA Collapsed SAH Cost %f\n", h_costs[0] / root_area, root_area);

    if (ads->use_persistent_kernel) lbvh_cuda_blas_prepare_persistent(ads, blas, object);

    cu_error = cudaMemcpy(h_bvh, thrust::raw_pointer_cast(d_bvh), blas->node_count * sizeof(*h_bvh), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_collapsed_bvh, thrust::raw_pointer_cast(d_collapsed_bvh), blas->node_count * sizeof(*h_collapsed_bvh), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_bvh_packed, blas->bvh_packed, blas->node_count * sizeof(*h_bvh_packed), cudaMemcpyDeviceToHost);

    kr_cuda_free((void**)&blas->transformations);
    kr_cuda_free((void**)&blas->max_proj);
    kr_cuda_free((void**)&blas->min_proj);
    kr_cuda_free((void**)&blas->visit_table);
    kr_cuda_free((void**)&blas->minmax_pairs);
    kr_cuda_free((void**)&blas->collapse_table);
    kr_cuda_free((void**)&blas->bvh);
    kr_cuda_free((void**)&blas->collapsed_bvh);
    kr_cuda_free((void**)&blas->obb_transforms);
    kr_cuda_free((void**)&blas->obbs);
    kr_cuda_free((void**)&blas->parents);
    kr_cuda_free((void**)&blas->costs);
    kr_cuda_free((void**)&blas->primitive_counts);
    kr_cuda_free((void**)&blas->mortons);

    return kr_success;
}

#define KR_ERROR_EMPTY_SCENE ((kr_error)"The scene has no intersectable primitives")
kr_error
lbvh_cuda_commit(kr_ads_lbvh_cuda* ads, kr_scene* scene) {
  ads->scene = scene;

    kr_size object_count = scene->object_count;
	kr_size instance_count = scene->instance_count;
	kr_size primitive_count = 0;

	if (instance_count > 0) {
		for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
			kr_object_instance* instance = &scene->instances[instance_index];
			kr_object* object = &scene->objects[instance->object_id];

            switch (object->type) {
            case KR_OBJECT_AABB:
                primitive_count += 1;
                break;
            case KR_OBJECT_MESH:
                primitive_count += (kr_u32)object->as_mesh.face_count;
                break;
            default:
                break;
            }
		}
	}
	else {
		for (kr_size object_index = 0; object_index < object_count; object_index++) {
			kr_object* object = &scene->objects[object_index];

			kr_u32 face_count = (kr_u32)object->as_mesh.face_count;
			primitive_count += face_count;
		}
	}

	if (primitive_count == 0) {
		return KR_ERROR_EMPTY_SCENE;
	}

    kr_ads_build_metrics total_metrics{};

    if (instance_count > 0) {
        for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
            kr_object_instance* instance = &scene->instances[instance_index];
            kr_object* object = &scene->objects[instance->object_id];

            lbvh_cuda_blas_commit(ads, &ads->blas, object, &total_metrics);
        }
    }
    else {
        for (kr_size object_index = 0; object_index < object_count; object_index++) {
            kr_object* object = &scene->objects[object_index];
            lbvh_cuda_blas_commit(ads, &ads->blas, object, &total_metrics);
        }
    }

    total_metrics.total_initial_tree_build =
        total_metrics.morton_calc_time +
        total_metrics.sort_time +
        total_metrics.radix_tree_time +
        total_metrics.aabb_calc_time;

    total_metrics.total_collapse_build =
        total_metrics.collapse_metrics.collapse_mark_time +
        total_metrics.collapse_metrics.collapse_time;

    total_metrics.total_obb_build =
        total_metrics.obb_metrics.obb_projection_time +
        total_metrics.obb_metrics.obb_candidates_eval_time +
        total_metrics.obb_metrics.obb_refit_time +
        total_metrics.obb_metrics.obb_finalize_time;

    total_metrics.total_time =
        total_metrics.total_initial_tree_build +
        total_metrics.total_obb_build +
        total_metrics.total_collapse_build;
       
    if (scene->benchmark_file) {
        FILE* out = fopen(scene->benchmark_file, "a");
        fprintf(out, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
            total_metrics.morton_calc_time,
            total_metrics.sort_time,
            total_metrics.radix_tree_time,
            total_metrics.aabb_calc_time,
            total_metrics.collapse_metrics.collapse_mark_time,
            total_metrics.collapse_metrics.collapse_time,
            total_metrics.obb_metrics.obb_projection_time,
            total_metrics.obb_metrics.obb_candidates_eval_time,
            total_metrics.obb_metrics.obb_refit_time,
            total_metrics.obb_metrics.obb_finalize_time,
            total_metrics.total_initial_tree_build,
            total_metrics.total_obb_build,
            total_metrics.total_collapse_build,
            total_metrics.total_time);
        fclose(out);
    }

  return kr_success;
}

kr_internal kr_inline int ConvertSMVer2Cores(int major, int minor) {
    const std::map<int, int> gpu_arch_cores_per_SM = {
        {0x10, 8},   {0x11, 8},   {0x12, 8},   {0x13, 8},   {0x20, 32},
        {0x21, 48},  {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128},
        {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
        {0x86, 128}, {0x87, 128} };
    const auto arch_version = (major << 4) + minor;
    const auto found = gpu_arch_cores_per_SM.find(arch_version);
    if (found == gpu_arch_cores_per_SM.end()) {
        return -1;
    }
    return found->second;
}

kr_error
lbvh_cuda_init(kr_ads_lbvh_cuda* ads, kr_descriptor_container* settings) {
    for (kr_size i = 0; i < settings->descriptor_count; i++) {
        if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "save_construction_information")) {
            ads->save_construction_information = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "collapse")) {
            ads->collapse = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "ct")) {
            ads->cost_traversal = (kr_scalar)atof(settings->descriptors[i].value);
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "ci")) {
            ads->cost_internal = (kr_scalar)atof(settings->descriptors[i].value);
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "use_obbs")) {
            ads->use_obbs = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "intersect_bounds_level")) {
            ads->intersect_bounds_level = atoi(settings->descriptors[i].value);
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "intersect_bounds")) {
            ads->intersect_bounds = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "use_persistent_kernel")) {
            ads->use_persistent_kernel = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int blocksPerSM = deviceProp.maxBlocksPerMultiProcessor;
    int warpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
    int warpsPerBlock = warpsPerSM / blocksPerSM;

    ads->cu_blockSize = warpsPerBlock * deviceProp.warpSize;
    ads->cu_gridSize = blocksPerSM * deviceProp.multiProcessorCount;
    ads->cu_core_count = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

    return kr_success;
}