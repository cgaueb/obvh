#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "ads/common/cuda/transforms.cuh"
#include "common/cuda/atomics.h"
#include "common/cuda/util.cuh"
#include "common/cuda/util.h"

#include <stdio.h>
#include <stack>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "common/logger.h" 

kr_internal kr_device kr_error
bvh_node_collapse(const kr_bvh_node* nodes, const kr_bvh_node* node, u32* leaves, u32 leaf_count, i32* position) {
    leaves[leaf_count - 1] = node->left;
    leaves[leaf_count - 2] = node->right;
    int stackSize = 2;

    u32 index;
    while (stackSize > 0)
    {
        // Pop stack
        index = leaves[leaf_count - stackSize];
        const kr_bvh_node* node = &nodes[index];
        --stackSize;

        if (node->nPrimitives > 0)
        {
            //int dataIndex = ads->primitives[index - ads->internal_count];
            int dataIndex = node->primitivesOffset;

            // Check if triangle is not already on the leaf
            b32 unique = kr_true;
            for (int i = 0; i < *position; ++i)
            {
                if (leaves[i] == dataIndex)
                {
                    unique = kr_false;
                    break;
                }
            }

            if (unique)
            {
                leaves[(*position)++] = dataIndex;
            }
        }
        else
        {
            i32 left = node->left;
            i32 right = node->right;
            leaves[leaf_count - 1 - stackSize++] = left;
            leaves[leaf_count - 1 - stackSize++] = right;
        }
    }

    return kr_success;
}

#define MAX_LEAF_SIZE 8
kr_internal kr_global
void bvh_mark_collapse_kernel(
    const kr_bvh_node* bvh,
    const u32* parents,
    kr_bvh_node* collapsed_bvh,
    b32* visit_table,
    b32* collapse_table,
    kr_scalar* costs,
    u32* primitive_counts,
    kr_scalar ci, kr_scalar ct,
    u32 leaf_count,
    u32 internal_count
) {
    const i32 leaf_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_index >= leaf_count)
        return;

    i32 i = leaf_index;

    const kr_bvh_node* internal_nodes = bvh;
    const kr_bvh_node* leaf_nodes = bvh + internal_count;

    const u32* internal_parents = parents;
    const u32* leaf_parents = parents + internal_count;

    kr_scalar* internal_costs = costs;
    kr_scalar* leaf_costs = costs + internal_count;

    b32* internal_collapse_table = collapse_table;
    b32* leaf_collapse_table = collapse_table + internal_count;

    leaf_costs[i] = ct * kr_aabb_surface_area3(leaf_nodes[i].bounds);

    const kr_bvh_node* parent;
    kr_u32 parent_index = leaf_parents[i];

    while (parent_index != 0xFFFFFFFF) {
        parent = &internal_nodes[parent_index];

        b32 visited = atomicExch(&visit_table[parent_index], kr_true);
        __threadfence();
        if (kr_false == visited) {
            break;
        }

        u32 count = primitive_counts[parent->left] + primitive_counts[parent->right];
        primitive_counts[parent_index] = count;

        kr_scalar area = kr_aabb_surface_area3(parent->bounds);
        kr_scalar node_cost = ci * area + costs[parent->left] + costs[parent->right];
        kr_scalar leaf_cost = ct * area * count;
        if (leaf_cost < node_cost && count <= MAX_LEAF_SIZE) {
            /* If a node is marked for collapse, 
             * we make sure that its children are also marked for collapse 
             */
            costs[parent_index] = leaf_cost;
            collapse_table[parent_index] = kr_true;
            collapse_table[parent->left] = kr_true;
            collapse_table[parent->right] = kr_true;

            /* Useful when we are collapsing nodes with
            leaf children */
            collapsed_bvh[parent->left].nPrimitives = 0xFFFF;
            collapsed_bvh[parent->right].nPrimitives = 0xFFFF;
        }
        else {
            costs[parent_index] = node_cost;
        }
        parent_index = parents[parent_index];
    }
}

kr_internal kr_global
void bvh_collapse_kernel(
    const kr_bvh_node* nodes,
    const u32* parents,
    const b32* collapse_table,
    kr_bvh_node* collapsed_nodes,
    b32* visit_table,
    u32* primitive_counts,
    u32* primitives,
    kr_scalar ci, kr_scalar ct,
    u32 leaf_count, u32 internal_count,
    u32* primitive_offset
) {
    const i32 leaf_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_index >= leaf_count)
        return;

    i32 i = leaf_index;

    const kr_bvh_node* internal_nodes = nodes;
    const kr_bvh_node* leaf_nodes = nodes + internal_count;

    const u32* internal_parents = parents;
    const u32* leaf_parents = parents + internal_count;

    const b32* internal_collapse_table = collapse_table;
    const b32* leaf_collapse_table = collapse_table + internal_count;

    u32 parent_index = leaf_parents[i];
    while (parent_index != 0xFFFFFFFF) {
        const kr_bvh_node* parent = &internal_nodes[parent_index];

        b32 visited = atomicExch(&visit_table[parent_index], kr_true);
        __threadfence();
        if (kr_false == visited) {
            break;
        }

        if (collapse_table[parent_index] == kr_true) {
            parent_index = parents[parent_index];
            continue;
        }

        u32 left_count = primitive_counts[parent->left];
        u32 right_count = primitive_counts[parent->right];

        b32 collapse_left = collapse_table[parent->left];
        b32 collapse_right = collapse_table[parent->right];

        if (left_count > 1 && collapse_left) {
            u32 offset = atomicAdd(primitive_offset, left_count);
            u32* buffer = primitives + offset;
            i32 position = 0;
            bvh_node_collapse(nodes, &nodes[parent->left], buffer, left_count, &position);
            primitive_counts[parent->left] = position;

            collapsed_nodes[parent->left].nPrimitives = position;
            collapsed_nodes[parent->left].primitivesOffset = offset;
        }

        if (right_count > 1 && collapse_right) {
            u32 offset = atomicAdd(primitive_offset, right_count);
            u32* buffer = primitives + offset;
            i32 position = 0;
            bvh_node_collapse(nodes, &nodes[parent->right], buffer, right_count, &position);
            primitive_counts[parent->right] = position;

            collapsed_nodes[parent->right].nPrimitives = position;
            collapsed_nodes[parent->right].primitivesOffset = offset;
        }

        parent_index = parents[parent_index];
    }
}

typedef kr_aabb3 kr_minmax_pair;


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

kr_internal __global__ void
calculate_obbs_kernel_shared(
    kr_bvh_node* nodes,
    const u32* parents, b32* visit_table,
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    kr_obb3* obbs,
    kr_scalar* global_min_proj, kr_scalar* global_max_proj,
    kr_minmax_pair* global_minmax_pairs, 
    u32 primitive_count,
    u32 internal_count) {

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
        cuvec4 face = faces[primitive_id];
        cvec3 v0 = vertices[face.x];
        cvec3 v1 = vertices[face.y];
        cvec3 v2 = vertices[face.z];
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

kr_inline_host_device
kr_scalar lbvh_dito14_obb_quality(cvec3 len) {
    return len.x * len.y + len.x * len.z + len.y * len.z; //half box area
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
kr_internal kr_inline_device kr_error
lbvh_dito14_furthest_point_pair(
    const kr_minmax_pair* minmax_pairs,
    vec3* p0, vec3* p1) {
    int at = 0;
    kr_scalar dist2, max_dist2;
    max_dist2 = kr_vdistance3sqr(minmax_pairs[0].min, minmax_pairs[0].max);
    for (int k = 1; k < K; k++)
    {
        dist2 = kr_vdistance3sqr(minmax_pairs[k].min, minmax_pairs[k].max);
        if (dist2 > max_dist2) { max_dist2 = dist2; at = k; }
    }

    *p0 = minmax_pairs[at].min;
    *p1 = minmax_pairs[at].max;

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
    for (int i = 1; i < point_count; i++)
    {
        dist2 = kr_vdistance_to_inf_edge3(points[i], p0, e0);
        if (dist2 > max_dist2)
        {
            max_dist2 = dist2;
            at = i;
        }
    }

    *p = points[at];

    return max_dist2;
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

kr_internal __global__ void
calculate_candidate_obb_kernel(
    kr_obb3* obbs,
    kr_scalar* min_proj, kr_scalar* max_proj,
    const kr_minmax_pair* minmax_pairs,
    u32 primitive_count, u32 internal_count) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_index >= internal_count)
        return;
   
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


kr_internal __global__ void
refit_obbs_kernel_new(
    const kr_bvh_node* nodes,
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    kr_obb3* obbs, kr_mat4* transforms,
    const u32* parents, const u32* primitive_counts,
    u32* processed_table, b32* visit_table,
    kr_scalar* min_proj, kr_scalar* max_proj,
    u32 primitive_count, u32 internal_count
) {
    // TODO 
    //* switch to CUDA fminf/fmaxf
    //* investigate early exit options

    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
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
    cuvec4 face = faces[primitive_id];
    cvec3 v0 = vertices[face.x];
    cvec3 v1 = vertices[face.y];
    cvec3 v2 = vertices[face.z];

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

    kr_mat4 s = kr_mscale4({ obb->ext.x * 2.0f, obb->ext.y * 2.0f, obb->ext.z * 2.0f });
    kr_mat4 t = kr_mtranslate4(obb->mid);
    *transform = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));

    parent_index = leaf_parents[i];
    while (parent_index != 0xFFFFFFFF) {
        parent = &internal_nodes[parent_index];

        kr_obb3* obb = &obbs[parent_index];
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

kr_internal
__global__ void
finalize_obbs_kernel(
    const kr_bvh_node* nodes,
    const kr_scalar* min_proj, const kr_scalar* max_proj,
    kr_obb3* obbs, kr_mat4* transforms,
    u32 primitive_count, u32 internal_count
) {
    // TODO 

    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_index >= internal_count)
        return;

    kr_obb3* obb = &obbs[node_index];
    kr_mat4* transform = &transforms[node_index];
    const kr_scalar* o_min_proj = &min_proj[7 * node_index];
    const kr_scalar* o_max_proj = &max_proj[7 * node_index];

    cvec3 aabb_len = { o_max_proj[0] - o_min_proj[0], o_max_proj[1] - o_min_proj[1], o_max_proj[2] - o_min_proj[2] };
    cvec3 obb_len = { o_max_proj[3] - o_min_proj[3], o_max_proj[4] - o_min_proj[4], o_max_proj[5] - o_min_proj[5] };

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
    u32 primitive_threshold) {
    cudaError_t cu_error;
    u32 node_count = primitive_count + internal_count;
    constexpr auto K = 7;

    b32* visit_table = (b32*)kr_cuda_allocate(internal_count * sizeof(*visit_table));
    kr_scalar* min_proj = (kr_scalar*)kr_cuda_allocate(K * node_count * sizeof(*min_proj));
    kr_scalar* max_proj = (kr_scalar*)kr_cuda_allocate(K * node_count * sizeof(*max_proj));
    kr_minmax_pair* minmax_pairs = (kr_minmax_pair*)kr_cuda_allocate(K * node_count * sizeof(*minmax_pairs));

    cu_error = cudaMemset(visit_table, 0xFFFFFFFF, internal_count * sizeof(*visit_table));

    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(32);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        const size_t sizeMinProj = sizeof(kr_scalar) * K;
        const size_t sizeMaxProj = sizeof(kr_scalar) * K;
        const size_t sizeMinMaxPair = sizeof(kr_minmax_pair) * K;

        const size_t cacheSize = blockSize.x * (
            sizeMinProj + sizeMaxProj + sizeMinMaxPair);

        calculate_obbs_kernel_shared << <gridSize, blockSize, cacheSize >> > (
            nodes,
            parents, visit_table,
            vertices, faces,
            obbs,
            min_proj, max_proj, minmax_pairs, 
            primitive_count, internal_count);
    });
    kr_log("Projection calculation took %fms\n", elapsed_ms);

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(32);
        int bx = (internal_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        calculate_candidate_obb_kernel <<< gridSize, blockSize >>> (
            obbs,
            min_proj, max_proj,
            minmax_pairs,
            primitive_count, internal_count);
        });
    kr_log("OBB candidates calculation took %fms\n", elapsed_ms);
    cu_error = cudaMemset(visit_table, 0, internal_count * sizeof(*visit_table));

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(128);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        refit_obbs_kernel_new <<< gridSize, blockSize >>> (
            nodes,
            vertices, faces,
            obbs, transforms,
            parents, primitive_counts,
            kr_null, visit_table,
            min_proj, max_proj,
            primitive_count, internal_count);
        });
    kr_log("OBB refit step new took %fms\n", elapsed_ms);
    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(512);
        int bx = (internal_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        finalize_obbs_kernel << < gridSize, blockSize >> > (
            nodes,
            min_proj, max_proj,
            obbs, transforms,
            primitive_count, internal_count);
        });
    kr_log("OBB finalize step new took %fms\n", elapsed_ms);
    cu_error = cudaDeviceSynchronize();

    kr_cuda_free((void**)&min_proj);
    kr_cuda_free((void**)&max_proj);
    kr_cuda_free((void**)&minmax_pairs);
    kr_cuda_free((void**)&visit_table);

    return kr_success;
}

kr_error kr_cuda_bvh_collapse(
    const kr_bvh_node* bvh,
    const u32* parents,
    kr_bvh_node* collapsed_bvh,
    u32* primitive_counts,
    u32* primitives,
    kr_scalar* costs,
    kr_scalar ci, kr_scalar ct,
    u32 leaf_count, u32 internal_count,
    kr_ads_collapse_metrics* metrics) {

    cudaError_t cu_error;
    u32 node_count = leaf_count + internal_count;

    b32* visit_table = (b32*)kr_cuda_allocate(internal_count * sizeof(*visit_table));
    b32* collapse_table = (b32*)kr_cuda_allocate(node_count * sizeof(*collapse_table));
    u32* primitive_offset = (kr_u32*)kr_cuda_allocate(sizeof(*primitive_offset));
    
    cu_error = cudaMemset(collapse_table, 0, node_count * sizeof(*collapse_table));
    cu_error = cudaMemset(visit_table, 0, internal_count * sizeof(*collapse_table));
    cu_error = cudaMemset(primitive_offset, 0, sizeof(*primitive_offset));

    thrust::copy(thrust::device, bvh, bvh + node_count, collapsed_bvh);

    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(256);
        int bx = ((kr_u32)leaf_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        bvh_mark_collapse_kernel << < gridSize, blockSize >> > (
            bvh, parents, collapsed_bvh,
            visit_table, collapse_table,
            costs,
            primitive_counts,
            ci, ct,
            leaf_count, internal_count);
    });

    if(metrics) metrics->collapse_mark_time += elapsed_ms;
    printf("Mark collapse calculation took %fms\n", elapsed_ms);

    cu_error = cudaMemset(visit_table, 0, internal_count * sizeof(*collapse_table));

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(256);
        int bx = ((kr_u32)leaf_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        bvh_collapse_kernel << < gridSize, blockSize >> > (
            bvh, parents, collapse_table, collapsed_bvh,
            visit_table,
            primitive_counts, primitives,
            ci, ct,
            leaf_count, internal_count,
            primitive_offset);
    });

    if (metrics) metrics->collapse_time += elapsed_ms;
    printf("Collapse calculation took %fms\n", elapsed_ms);

    kr_cuda_free((void**)&visit_table);
    kr_cuda_free((void**)&collapse_table);
    kr_cuda_free((void**)&primitive_offset);

    return kr_success;
}

template<bool use_centroids = false, int BLOCK_SIZE = 1>
kr_internal kr_global
void bounding_box_kernel_atomics(
    cuvec4* faces,
    cvec3* vertices,
    u32 face_count,
    kr_aabb3* aabb
) {
    const u32 primitive_index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float cache[3 * BLOCK_SIZE];
    vec3* bound = (vec3*)cache;
    bound[threadIdx.x] = vertices[0];
    __syncthreads();

    if (primitive_index >= face_count)
        return;

    aabb3 block_aabb = kr_aabb_empty3();
    aabb3 face_aabb = kr_aabb_empty3();

    cuvec4 face = faces[primitive_index];
    cvec3  va   = vertices[face.x];
    cvec3  vb   = vertices[face.y];
    cvec3  vc   = vertices[face.z];

    face_aabb = kr_aabb_expand3(face_aabb, va);
    face_aabb = kr_aabb_expand3(face_aabb, vb);
    face_aabb = kr_aabb_expand3(face_aabb, vc);

    if constexpr (use_centroids) {
        block_aabb = kr_aabb_expand3(block_aabb, kr_aabb_center3(face_aabb));
    } else {
        block_aabb = face_aabb;
    }

    // Perform parallel reduction within the block.
    bound[threadIdx.x] = block_aabb.min;
    bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 1]);
    bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 2]);
    bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 4]);
    bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 8]);
    bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) bound[threadIdx.x] = kr_vmin3(bound[threadIdx.x], bound[threadIdx.x ^ 128]);

    // Update global bounding box.
    if (threadIdx.x == 0) {
        atomicMin(&aabb->min.x, bound[threadIdx.x].x);
        atomicMin(&aabb->min.y, bound[threadIdx.x].y);
        atomicMin(&aabb->min.z, bound[threadIdx.x].z);
    }

    __syncthreads();

    bound[threadIdx.x] = block_aabb.max;
    bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 1]);
    bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 2]);
    bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 4]);
    bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 8]);
    bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) bound[threadIdx.x] = kr_vmax3(bound[threadIdx.x], bound[threadIdx.x ^ 128]);


    // Update global bounding box.
    if (threadIdx.x == 0) {
        atomicMax(&aabb->max.x, bound[threadIdx.x].x);
        atomicMax(&aabb->max.y, bound[threadIdx.x].y);
        atomicMax(&aabb->max.z, bound[threadIdx.x].z);
    }
}

kr_error kr_cuda_bbox_calculate(
    cuvec4* faces,
    cvec3* vertices,
    u32 face_count,
    kr_aabb3* d_aabb
) {
    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        constexpr auto block_size = 256;
        dim3 blockSize = dim3(block_size);
        int bx = ((kr_u32)face_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        bounding_box_kernel_atomics<false, block_size> <<< gridSize, blockSize >>> (
            faces, vertices, face_count, d_aabb);
    });
    printf("AABB calculation took %fms\n", elapsed_ms);

    return kr_success;
}

kr_error kr_cuda_centroid_bbox_calculate(
    cuvec4* faces,
    cvec3* vertices,
    u32 face_count,
    kr_aabb3* d_aabb
) {
    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        constexpr auto block_size = 256;
        dim3 blockSize = dim3(block_size);
        int bx = ((kr_u32)face_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        bounding_box_kernel_atomics<true, block_size> << < gridSize, blockSize >> > (
            faces, vertices, face_count, d_aabb);
        });
    printf("Centroid AABB calculation took %fms\n", elapsed_ms);

    return kr_success;
}

kr_internal kr_global void
bvh_persistent_prepare(
    const kr_bvh_node* bvh,
    kr_bvh_node_packed* packed_bvh,
    u32 node_count) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_index >= node_count)
        return;

    const kr_bvh_node* node = &bvh[node_index];
    kr_bvh_node_packed* packed_node = &packed_bvh[node_index];

    if (node->nPrimitives == 0) {
        const kr_bvh_node* l = &bvh[node->left];
        const kr_bvh_node* r = &bvh[node->right];

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
}

kr_error kr_cuda_bvh_persistent(
    const kr_bvh_node* bvh,
    kr_bvh_node_packed* packed_bvh,
    u32 node_count
) {
    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(512);
        int bx = (node_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        bvh_persistent_prepare <<< gridSize, blockSize >>> (
            bvh,
            packed_bvh,
            node_count);
        });
    //kr_log("Persistent calculation took %fms\n", elapsed_ms);

    return kr_success;
}

kr_internal kr_global void
obvh_persistent_prepare(
    const kr_bvh_node* bvh,
    const kr_mat4* transforms,
    kr_bvh_node_packed* packed_bvh,
    kr_bvh_transformation_pair* packed_transforms,
    u32 node_count) {
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_index >= node_count)
        return;

    const kr_bvh_node* node = &bvh[node_index];

    if (node->nPrimitives == 0xFFFF)
        return;

    kr_bvh_node_packed* packed_node = &packed_bvh[node_index];
    kr_bvh_transformation_pair* packed_transform = &packed_transforms[node_index];

    packed_node->metadata.z = node->axis;

    if (node->nPrimitives == 0) {
        const kr_bvh_node* l = &bvh[node->left];
        const kr_bvh_node* r = &bvh[node->right];

        packed_node->lbounds = l->bounds;
        packed_node->rbounds = r->bounds;
        packed_node->left = (l->nPrimitives > 0) ? ~((i32)node->left) : (i32)node->left;
        packed_node->right = (r->nPrimitives > 0) ? ~((i32)node->right) : (i32)node->right;
    
        //packed_transform->l = kr_m43from4(transforms[node->left]);
        //packed_transform->r = kr_m43from4(transforms[node->right]);
        
        packed_transform->l = transforms[node->left];
        packed_transform->r = transforms[node->right];
    }
    else {
        packed_node->lbounds = node->bounds;
        packed_node->left = (i32)node->primitivesOffset;
        packed_node->right = (i32)(node->primitivesOffset + node->nPrimitives);
    }
    packed_node->metadata.z = node->axis;
}

kr_error kr_cuda_obvh_persistent(
    const kr_bvh_node* bvh,
    const kr_mat4* transforms,
    kr_bvh_node_packed* packed_bvh,
    kr_bvh_transformation_pair* packed_transforms,
    u32 node_count
) {
    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(512);
        int bx = (node_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        obvh_persistent_prepare << < gridSize, blockSize >> > (
            bvh,
            transforms,
            packed_bvh,
            packed_transforms,
            node_count);
        });
    //kr_log("Persistent calculation took %fms\n", elapsed_ms);

    return kr_success;
}

kr_error kr_cuda_soa_obvh(
    const kr_bvh_node_packed* packed_bvh,
    const kr_bvh_transformation_pair* packed_transforms,
    kr_SoA_OBVH*& soa_bvh,
    u32 internal_count,
    u32 node_count)
{
    mat43* left_T = new mat43[internal_count];
    mat43* right_T = new mat43[internal_count];
    ivec4* metadata = new ivec4[node_count];

    kr_bvh_node_packed* h_bvh = new kr_bvh_node_packed[node_count];
    kr_bvh_transformation_pair* h_transforms = new kr_bvh_transformation_pair[internal_count];

    cudaMemcpy(h_bvh, packed_bvh, sizeof(kr_bvh_node_packed) * node_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_transforms, packed_transforms, sizeof(kr_bvh_transformation_pair) * internal_count, cudaMemcpyDeviceToHost);

    for (int node_i = 0; node_i < internal_count; ++node_i)
    {
        left_T[node_i] = kr_m43from4(h_transforms[node_i].l);
        right_T[node_i] = kr_m43from4(h_transforms[node_i].r);
        //left_T[node_i] = h_transforms[node_i].l;
        //right_T[node_i] = h_transforms[node_i].r;
    }

    for (int node_i = 0; node_i < node_count; ++node_i)
    {
        metadata[node_i] = h_bvh[node_i].metadata;
    }

#if 0
    std::stack<std::tuple<int, int, int>> tree_stack;
    tree_stack.push({ 0, 0, -1 });
    int max_aabb_lvl = 1;

    while (!tree_stack.empty())
    {
        int node_i = std::get<0>(tree_stack.top());
        int node_i_lvl = std::get<1>(tree_stack.top());
        int parent = std::get<2>(tree_stack.top());
        tree_stack.pop();

        if (node_i_lvl <= max_aabb_lvl)
        {
            ivec4 meta = metadata[node_i];

            if (parent > -1) { metadata[parent].z = 0; }
            left_T[node_i].cols[0] = h_bvh[node_i].lbounds.min;
            left_T[node_i].cols[1] = h_bvh[node_i].lbounds.max;
            right_T[node_i].cols[0] = h_bvh[node_i].rbounds.min;
            right_T[node_i].cols[1] = h_bvh[node_i].rbounds.max;

            int next_lvl = node_i_lvl + 1;
            if (meta.x > 0) tree_stack.push({ meta.x, next_lvl, node_i });
            if (meta.y > 0) tree_stack.push({ meta.y, next_lvl, node_i });
        }
    }
#endif
    /*
    //HACK
    left_T[0].cols[0] = h_bvh[0].lbounds.min;
    left_T[0].cols[1] = h_bvh[0].lbounds.max;
    right_T[0].cols[0] = h_bvh[0].rbounds.min;
    right_T[0].cols[1] = h_bvh[0].rbounds.max;*/

    delete[] h_bvh, h_transforms;

    kr_SoA_OBVH* h_soa_obvh = new kr_SoA_OBVH();
    h_soa_obvh->left_T = (mat43*)kr_cuda_allocate(internal_count * sizeof(mat43));
    h_soa_obvh->right_T = (mat43*)kr_cuda_allocate(internal_count * sizeof(mat43));
    h_soa_obvh->metadata = (ivec4*)kr_cuda_allocate(node_count * sizeof(ivec4));

    cudaMemcpy(h_soa_obvh->left_T, left_T, sizeof(mat43) * internal_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_obvh->right_T, right_T, sizeof(mat43) * internal_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_obvh->metadata, metadata, sizeof(ivec4) * node_count, cudaMemcpyHostToDevice);

    delete[] left_T, right_T, metadata;

    soa_bvh = (kr_SoA_OBVH*)kr_cuda_allocate(sizeof(kr_SoA_OBVH));
    cudaMemcpy(soa_bvh, h_soa_obvh, sizeof(kr_SoA_OBVH), cudaMemcpyHostToDevice);
    delete h_soa_obvh;

    return kr_success;
}


kr_error kr_cuda_soa_compressed_obvh(
    const kr_bvh_node_packed* packed_bvh,
    const kr_bvh_transformation_pair* packed_transforms,
    kr_SoA_OBVH_Compressed*& soa_bvh,
    u32 internal_count,
    u32 node_count) {
    aabb3* left_T = new aabb3[internal_count];
    aabb3* right_T = new aabb3[internal_count];
    ivec4* metadata = new ivec4[node_count];

    kr_bvh_node_packed* h_bvh = new kr_bvh_node_packed[node_count];
    kr_bvh_transformation_pair* h_transforms = new kr_bvh_transformation_pair[internal_count];

    cudaMemcpy(h_bvh, packed_bvh, sizeof(kr_bvh_node_packed) * node_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_transforms, packed_transforms, sizeof(kr_bvh_transformation_pair) * internal_count, cudaMemcpyDeviceToHost);


    for (int node_i = 0; node_i < node_count; ++node_i)
    {
        metadata[node_i] = h_bvh[node_i].metadata;
    }

    for (int node_i = 0; node_i < internal_count; ++node_i)
    {
        left_T[node_i].min = { h_transforms[node_i].l.cols[1].x, h_transforms[node_i].l.cols[1].y, h_transforms[node_i].l.cols[1].z };
        left_T[node_i].max = { 1.0f / (h_transforms[node_i].l.cols[2].x), 1.0f / (h_transforms[node_i].l.cols[2].y), 1.0f / (h_transforms[node_i].l.cols[2].z) };
        right_T[node_i].min = { h_transforms[node_i].r.cols[1].x, h_transforms[node_i].r.cols[1].y, h_transforms[node_i].r.cols[1].z };
        right_T[node_i].max = { 1.0f / (h_transforms[node_i].r.cols[2].x), 1.0f / (h_transforms[node_i].r.cols[2].y), 1.0f / (h_transforms[node_i].r.cols[2].z) };

        i8 qx = -i8(h_transforms[node_i].l.cols[0].x * 127.0f);
        i8 qy = -i8(h_transforms[node_i].l.cols[0].y * 127.0f);
        i8 qz = -i8(h_transforms[node_i].l.cols[0].z * 127.0f);
        i8 qw = i8(h_transforms[node_i].l.cols[0].w * 127.0f);

        metadata[node_i].z |= (u8)qx;
        metadata[node_i].z <<= 8;
        metadata[node_i].z |= (u8)qy;
        metadata[node_i].z <<= 8;
        metadata[node_i].z |= (u8)qz;
        metadata[node_i].z <<= 8;
        metadata[node_i].z |= (u8)qw;

        kr_scalar qcx = (kr_scalar)char((metadata[node_i].z >> 24) & 0x000000FF) / 127.0f;
        kr_scalar qcy = (kr_scalar)char((metadata[node_i].z >> 16) & 0x000000FF) / 127.0f;
        kr_scalar qcz = (kr_scalar)char((metadata[node_i].z >> 8) & 0x000000FF) / 127.0f;
        kr_scalar qcw = (kr_scalar)char((metadata[node_i].z >> 0) & 0x000000FF) / 127.0f;

        kr_mat4 r = kr_mrotate4({ qcx, qcy, qcz }, qcw * KR_PI);
        //kr_mat4 t = kr_mtranslate4(lbounds.min);
        //kr_mat4 s = kr_mscale4(lbounds.max);
        //kr_mat4 trs = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));

        qx = -i8(h_transforms[node_i].r.cols[0].x * 127.0f);
        qy = -i8(h_transforms[node_i].r.cols[0].y * 127.0f);
        qz = -i8(h_transforms[node_i].r.cols[0].z * 127.0f);
        qw = i8(h_transforms[node_i].r.cols[0].w * 127.0f);

        metadata[node_i].w |= (u8)qx;
        metadata[node_i].w <<= 8;
        metadata[node_i].w |= (u8)qy;
        metadata[node_i].w <<= 8;
        metadata[node_i].w |= (u8)qz;
        metadata[node_i].w <<= 8;
        metadata[node_i].w |= (u8)qw;
    }

    kr_SoA_OBVH_Compressed* h_soa_obvh = new kr_SoA_OBVH_Compressed();

    h_soa_obvh->lbounds = (aabb3*)kr_cuda_allocate(internal_count * sizeof(aabb3));
    h_soa_obvh->rbounds = (aabb3*)kr_cuda_allocate(internal_count * sizeof(aabb3));
    h_soa_obvh->metadata = (ivec4*)kr_cuda_allocate(node_count * sizeof(ivec4));

    cudaMemcpy(h_soa_obvh->lbounds, left_T, sizeof(aabb3) * internal_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_obvh->rbounds, right_T, sizeof(aabb3) * internal_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_obvh->metadata, metadata, sizeof(ivec4) * node_count, cudaMemcpyHostToDevice);

    delete[] left_T, right_T, metadata;

    soa_bvh = (kr_SoA_OBVH_Compressed*)kr_cuda_allocate(sizeof(kr_SoA_OBVH_Compressed));
    cudaMemcpy(soa_bvh, h_soa_obvh, sizeof(kr_SoA_OBVH_Compressed), cudaMemcpyHostToDevice);
    delete h_soa_obvh;

    return kr_success;
}

kr_error kr_cuda_soa_bvh(
    const kr_bvh_node_packed* packed_bvh,
    kr_SoA_BVH*& soa_bvh,
    u32 node_count)
{
    vec4* lbbox_XY = new vec4[node_count];
    vec4* rbbox_XY = new vec4[node_count];
    vec4* lrbbox_Z = new vec4[node_count];
    ivec4* metadata = new ivec4[node_count];

    kr_bvh_node_packed* h_bvh = new kr_bvh_node_packed[node_count];

    cudaMemcpy(h_bvh, packed_bvh, sizeof(kr_bvh_node_packed) * node_count, cudaMemcpyDeviceToHost);

    for (int node_i = 0; node_i < node_count; ++node_i)
    {
        lbbox_XY[node_i] = {
            h_bvh[node_i].lbounds.min.x, h_bvh[node_i].lbounds.max.x,
            h_bvh[node_i].lbounds.min.y, h_bvh[node_i].lbounds.max.y };

        rbbox_XY[node_i] = {
            h_bvh[node_i].rbounds.min.x, h_bvh[node_i].rbounds.max.x,
            h_bvh[node_i].rbounds.min.y, h_bvh[node_i].rbounds.max.y };

        lrbbox_Z[node_i] = {
            h_bvh[node_i].lbounds.min.z, h_bvh[node_i].lbounds.max.z,
            h_bvh[node_i].rbounds.min.z, h_bvh[node_i].rbounds.max.z };

        metadata[node_i] = h_bvh[node_i].metadata;
    }

    delete[] h_bvh;

    kr_SoA_BVH* h_soa_bvh = new kr_SoA_BVH();
    h_soa_bvh->lbbox_XY = (vec4*)kr_cuda_allocate(node_count * sizeof(vec4));
    h_soa_bvh->rbbox_XY = (vec4*)kr_cuda_allocate(node_count * sizeof(vec4));
    h_soa_bvh->lrbbox_Z = (vec4*)kr_cuda_allocate(node_count * sizeof(vec4));
    h_soa_bvh->metadata = (ivec4*)kr_cuda_allocate(node_count * sizeof(ivec4));

    cudaMemcpy(h_soa_bvh->lbbox_XY, lbbox_XY, sizeof(vec4) * node_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_bvh->rbbox_XY, rbbox_XY, sizeof(vec4) * node_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_bvh->lrbbox_Z, lrbbox_Z, sizeof(vec4) * node_count, cudaMemcpyHostToDevice);
    cudaMemcpy(h_soa_bvh->metadata, metadata, sizeof(ivec4) * node_count, cudaMemcpyHostToDevice);

    delete[] lbbox_XY, rbbox_XY, lrbbox_Z, metadata;

    soa_bvh = (kr_SoA_BVH*)kr_cuda_allocate(sizeof(kr_SoA_BVH));
    cudaMemcpy(soa_bvh, h_soa_bvh, sizeof(kr_SoA_BVH), cudaMemcpyHostToDevice);
    delete h_soa_bvh;

    return kr_success;
}
