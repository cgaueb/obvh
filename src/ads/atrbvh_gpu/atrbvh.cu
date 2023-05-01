#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "common/korangar.h"
#include "common/ads.h"

#include "atrbvh.h"

#include "common/logger.h" 
#include "common/geometry.h" 
#include "common/util.h"
#include "common/cuda/util.h"
#include "common/cuda/atomics.h"

#include "ads/common/cuda/intersectors.cuh"
#include "ads/common/cuda/transforms.cuh"

#include "ads/common/transforms.h"

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
#include <stdio.h>

#include "SceneWrapper.h"
#include "LBVHBuilder.h"
#include "AgglomerativeTreeletOptimizer.h"
#include "BVHTreeInstanceManager.h"
#include "BVHTree.h"

kr_error
atrbvh_cuda_query_intersection(kr_ads_bvh2_cuda* ads, kr_intersection_query* query) {

    const kr_ads_blas_cuda* blas = &ads->blas;
    const kr_object_cu* object = blas->h_object_cu;

    thrust::fill(thrust::device, blas->ray_counter, blas->ray_counter + 1, 0);

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

    }

    return kr_success;
}

kr_ads_bvh2_cuda* atrbvh_cuda_create() { return (kr_ads_bvh2_cuda*)kr_allocate(sizeof(kr_ads_bvh2_cuda)); }

__device__ void copyBounds(vec3& bminOut, vec3& bmaxOut, float4& bminIn, float4& bmaxIn)
{
    bminOut.x = bminIn.x;
    bminOut.y = bminIn.y;
    bminOut.z = bminIn.z;

    bmaxOut.x = bmaxIn.x;
    bmaxOut.y = bmaxIn.y;
    bmaxOut.z = bmaxIn.z;
}

__global__ void copyTree_kernel(const BVHTree* deviceTree,
    kr_bvh_node* nodes, u32* parents, u32 primitive_count)
{
    const int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 internal_count = (u32)(primitive_count - 1);
    const u32 node_count = primitive_count + internal_count;
    if (node_index >= node_count) return;

    parents[node_index] = deviceTree->ParentIndex(node_index);
    kr_bvh_node& node = nodes[node_index];
    int primIdx = deviceTree->DataIndex(node_index);
    
    if (primIdx < 0) // internal
    {
        node.left = deviceTree->LeftIndex(node_index);
        node.right = deviceTree->RightIndex(node_index);
    }
    else
    {
        copyBounds(node.bounds.min, node.bounds.max,
            deviceTree->BoundingBoxMin(node_index),
            deviceTree->BoundingBoxMax(node_index));

        node.nPrimitives = 1;
        node.primitivesOffset = primIdx;
    }
}

__global__ void calculate_aabbs_kernel(
    kr_bvh_node* nodes,
    u32* parents,
    u32* primitive_counts,
    kr_scalar* costs,
    b32* visit_table,
    u32 primitive_count)
{
    const int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
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
atrbvh_cuda_blas_prepare_persistent(kr_ads_bvh2_cuda* ads, kr_ads_blas_cuda* blas, kr_object* object) {
    kr_bvh_node* bvh = (kr_null != blas->collapsed_bvh) ? blas->collapsed_bvh : blas->bvh;

    if (!ads->use_obbs) {
        kr_cuda_bvh_persistent(
            bvh,
            blas->bvh_packed,
            blas->node_count
        );

        kr_cuda_soa_bvh(
            blas->bvh_packed,
            blas->soa_bvh_packed,
            blas->node_count);
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

kr_internal kr_error atrbvh_cuda_blas_commit(kr_ads_bvh2_cuda* ads, kr_ads_blas_cuda* blas, kr_object* object)
{
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

    float4* vertices;

    switch (object->type) {
    case KR_OBJECT_AABB:
        primitive_count = 1;
        break;
    case KR_OBJECT_MESH: {
        kr_u32 face_count = (kr_u32)object->as_mesh.face_count;
        kr_u32 attr_count = (kr_u32)object->as_mesh.attr_count;
        primitive_count = (kr_u32)object->as_mesh.face_count;
        vertices = new float4[face_count * 3];

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

            vertices[3 * face_index] = { va.x, va.y, va.z, 0 };
            vertices[3 * face_index + 1] = { vb.x, vb.y, vb.z, 0 };
            vertices[3 * face_index + 2] = { vc.x, vc.y, vc.z, 0 };
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

    ads->aabb = scene_bbox;

    blas->node_count = 2 * (kr_u32)primitive_count - 1;
    blas->internal_count = (kr_u32)primitive_count - 1;
    blas->primitive_count = (kr_u32)primitive_count;
    blas->leaf_count = (kr_u32)primitive_count;

    float3 boundingBoxMin = { scene_bbox.min.x, scene_bbox.min.y, scene_bbox.min.z };
    float3 boundingBoxMax = { scene_bbox.max.x, scene_bbox.max.y, scene_bbox.max.z };
    SceneWrapper* wrapper = new SceneWrapper(primitive_count, vertices, boundingBoxMin, boundingBoxMax);

    float sah = 0.f;
    float time_lbvh = 0.f;
    float time_optimization = 0.f;

    LBVHBuilder builder(true);

    BVHTree* deviceTree = builder.BuildTree(wrapper,
        ads->cost_internal, ads->cost_traversal, nullptr, &time_lbvh);

    AgglomerativeTreeletOptimizer agglomerativeOptimizer(9, 2);

    agglomerativeOptimizer.Optimize(deviceTree, &time_optimization, &sah,
        ads->cost_internal, ads->cost_traversal);

    BVHTreeInstanceManager instanceManager;

    blas->primitive_counts = (u32*)kr_cuda_allocate(blas->node_count * sizeof(*blas->primitive_counts));
    blas->obb_transforms = (kr_mat4*)kr_cuda_allocate(blas->node_count * sizeof(*blas->obb_transforms));
    blas->obbs = (kr_obb3*)kr_cuda_allocate(blas->node_count * sizeof(*blas->obbs));
    blas->mortons = (kr_morton*)kr_cuda_allocate(blas->primitive_count * sizeof(*blas->mortons));
    blas->primitives = (kr_u32*)kr_cuda_allocate(blas->primitive_count * sizeof(*blas->primitives));
    blas->costs = (kr_scalar*)kr_cuda_allocate(blas->node_count * sizeof(*blas->costs));
    blas->parents = (u32*)kr_cuda_allocate(blas->node_count * sizeof(*blas->parents));
    blas->visit_table = (b32*)kr_cuda_allocate(blas->internal_count * sizeof(*blas->visit_table));
    blas->collapse_table = (b32*)kr_cuda_allocate(blas->node_count * sizeof(*blas->collapse_table));

    KR_CUDA_ALLOC_THRUST_DECLARE(kr_bvh_node, d_bvh, blas->node_count);

    cu_error = cudaMemset(blas->visit_table, 0, blas->internal_count * sizeof(*blas->visit_table));

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(256);
        int bx = (blas->node_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        copyTree_kernel << <gridSize, blockSize >> > (deviceTree,
            thrust::raw_pointer_cast(d_bvh), blas->parents, blas->leaf_count);
        });

    instanceManager.FreeDeviceTree(deviceTree);
    delete wrapper;

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(256);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        calculate_aabbs_kernel <<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_bvh),
            blas->parents,
            blas->primitive_counts,
            blas->costs,
            blas->visit_table,
            blas->primitive_count);
    });

    KR_CUDA_ALLOC_THRUST_DECLARE(u32, d_ray_counter, 1);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_bvh_node_packed, d_bvh_packed, blas->node_count);

    thrust::device_ptr<kr_bvh_node> d_collapsed_bvh = kr_null;

    if (kr_false != ads->collapse) {
        d_collapsed_bvh = thrust::device_ptr<kr_bvh_node>((kr_bvh_node*)kr_cuda_allocate(blas->node_count * sizeof(kr_bvh_node)));
    }

    blas->d_object_cu = d_object_cu;
    blas->h_object_cu = h_object_cu;
    blas->bvh = thrust::raw_pointer_cast(d_bvh);
    blas->collapsed_bvh = thrust::raw_pointer_cast(d_collapsed_bvh);
    blas->bvh_packed = thrust::raw_pointer_cast(d_bvh_packed);
    blas->ray_counter = thrust::raw_pointer_cast(d_ray_counter);

    kr_bvh_node* h_bvh = (kr_bvh_node*)kr_allocate(blas->node_count * sizeof(*h_bvh));
    cudaMemcpy(h_bvh, blas->bvh, blas->node_count * sizeof(*h_bvh), cudaMemcpyDeviceToHost);
    
    kr_log("AABB-ATRBVH SAH Cost %f\n", kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, kr_null, ads->scene->benchmark_file));

    if (kr_false != ads->use_obbs) {
        u32* h_primitive_counts = (u32*)kr_allocate(blas->node_count * sizeof(*h_primitive_counts));
        kr_mat4* h_obb_transforms = (kr_mat4*)kr_allocate(blas->node_count * sizeof(*h_obb_transforms));
        kr_obb3* h_obbs = (kr_obb3*)kr_allocate(blas->node_count * sizeof(*h_obbs));
        u32* h_primitives = (u32*)kr_allocate(blas->primitive_count * sizeof(*h_primitives));
        kr_scalar* h_costs = (kr_scalar*)kr_allocate(blas->node_count * sizeof(*h_costs));
        u32* h_parents = (u32*)kr_allocate(blas->node_count * sizeof(*h_parents));
        
        cudaMemcpy(h_costs, blas->costs, blas->node_count * sizeof(*h_costs), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_primitive_counts, blas->primitive_counts, blas->node_count * sizeof(*h_primitive_counts), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parents, blas->parents, blas->node_count * sizeof(*h_parents), cudaMemcpyDeviceToHost);

        kr_cuda_bvh_obb_tree(
            blas->bvh, blas->parents,
            blas->primitive_counts, blas->costs,
            h_object_cu->as_mesh.vertices,
            h_object_cu->as_mesh.faces,
            blas->primitives,
            blas->obbs,
            blas->obb_transforms,
            ads->cost_internal, ads->cost_traversal, 1.0f,
            blas->leaf_count, blas->internal_count, -1
        );
    }

    if (kr_false != ads->collapse)
    {
        kr_cuda_bvh_collapse(
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
            kr_null);

        cudaMemcpy(h_bvh, blas->collapsed_bvh, blas->node_count * sizeof(kr_bvh_node), cudaMemcpyDeviceToHost);
        kr_log("AABB-COLLAPSED-ATRBVH SAH Cost %f\n", kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, nullptr, ads->scene->benchmark_file));

        if (kr_false != ads->use_obbs) {
            kr_obb3* h_obbs = (kr_obb3*)kr_allocate(blas->node_count * sizeof(*h_obbs));
            cudaMemcpy(h_obbs, blas->obbs, blas->node_count * sizeof(kr_obb3), cudaMemcpyDeviceToHost);
            kr_log("OBB-COLLAPSED-ATRBVH SAH Cost %f\n", kr_bvh_sah(h_bvh, 0, ads->cost_internal, ads->cost_traversal, h_obbs, ads->scene->benchmark_file));
            kr_free(((void**)&h_obbs));
        }
        
    }

    if (ads->use_persistent_kernel) atrbvh_cuda_blas_prepare_persistent(ads, blas, object);

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

    cu_error = cudaDeviceSynchronize();

    return kr_success;
}

#define KR_ERROR_EMPTY_SCENE ((kr_error)"The scene has no intersectable primitives")
kr_error atrbvh_cuda_commit(kr_ads_bvh2_cuda* ads, kr_scene* scene) {
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

    if (instance_count > 0) {
        for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
            kr_object_instance* instance = &scene->instances[instance_index];
            kr_object* object = &scene->objects[instance->object_id];

            atrbvh_cuda_blas_commit(ads, &ads->blas, object);
        }
    }
    else {
        for (kr_size object_index = 0; object_index < object_count; object_index++) {
            kr_object* object = &scene->objects[object_index];
            atrbvh_cuda_blas_commit(ads, &ads->blas, object);
        }
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

kr_error atrbvh_cuda_init(kr_ads_bvh2_cuda* ads, kr_descriptor_container* settings) {
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