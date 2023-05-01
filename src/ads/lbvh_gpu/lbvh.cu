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

kr_internal kr_error
lbvh_cuda_calculate_obbs(kr_ads_lbvh_cuda* ads, kr_ads_obb_metrics * metrics) {
    kr_ads_blas_cuda* blas = &ads->blas;
    const kr_object_cu* object = blas->h_object_cu;
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