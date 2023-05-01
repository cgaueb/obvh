#define KR_VECMATH_IMPL
#include "common/vecmath.h"
#include "common/util.h"

#include "common/geometry.h"

#include "transforms.h"

#include <vector>
#include <list>
#include <stack>

kr_internal i32
kr_bvh_node_collapse(
    const kr_bvh_node* nodes, const kr_bvh_node* node, 
    u32* leaves, u32 leaf_count) {
    i32 position = 0;

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
            int dataIndex = node->primitivesOffset;

            // Check if triangle is not already on the leaf
            b32 unique = kr_true;
            for (int i = 0; i < position; ++i)
            {
                if (leaves[i] == dataIndex)
                {
                    unique = kr_false;
                    break;
                }
            }

            if (unique)
            {
                leaves[position++] = dataIndex;
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

    return position;
}

kr_error kr_bvh_collapse(
    const kr_bvh_node* nodes,
    const u32* parents,
    kr_bvh_node* collapsed_nodes,
    u32* primitive_counts,
    u32* primitives,
    kr_scalar* costs,
    kr_scalar ci, kr_scalar ct,
    u32 primitive_count,
    u32 internal_count,
    const kr_obb3* obbs) {
    const u32 node_count = internal_count + primitive_count;

    kr_memcpy(collapsed_nodes, nodes, node_count * sizeof(*collapsed_nodes));

    KR_ALLOC_DECLARE(b32, collapse_table, node_count);
    KR_ALLOC_DECLARE(b32, visit_table, internal_count);

    const kr_bvh_node* leaf_nodes = nodes + internal_count;
    const kr_bvh_node* internal_nodes = nodes;

    const u32* internal_parents = parents;
    const u32* leaf_parents = parents + internal_count;

    u32* leaf_primitive_counts = primitive_counts + internal_count;
    u32* internal_primitive_counts = primitive_counts;

    kr_scalar* leaf_costs = costs + internal_count;
    kr_scalar* internal_costs = costs;

    b32* leaf_collapse_table = collapse_table + internal_count;
    b32* internal_collapse_table = collapse_table;

    for (u32 i = 0; i < primitive_count; ++i) {
        kr_scalar area = obbs ? kr_obb_surface_area3(obbs[internal_count + i]) : kr_aabb_surface_area3(leaf_nodes[i].bounds);
        leaf_primitive_counts[i] = 1;
        leaf_costs[i] = ct * area;
        leaf_collapse_table[i] = kr_false;
    }

    for (u32 i = 0; i < primitive_count; ++i) {
        const kr_bvh_node* parent;
        u32 parent_index = leaf_parents[i];
        while (parent_index != 0xFFFFFFFF) {
            parent = &internal_nodes[parent_index];

            if (kr_false == visit_table[parent_index]) {
                visit_table[parent_index] = kr_true;
                break;
            }

            u32 count = primitive_counts[parent->left] + primitive_counts[parent->right];
            primitive_counts[parent_index] = count;

            kr_scalar area = obbs ? kr_obb_surface_area3(obbs[parent_index]) : kr_aabb_surface_area3(parent->bounds);
            costs[parent_index] = ((obbs ? 2 : 1) * ci) * area + costs[parent->left] + costs[parent->right];
            kr_scalar leaf_cost = ct * area * count;

#define MAX_LEAF_SIZE 8
            if (leaf_cost < costs[parent_index] && count <= MAX_LEAF_SIZE)
            {
                internal_costs[parent_index] = leaf_cost;
                internal_collapse_table[parent_index] = kr_true;

                /* Also mark those below for collapse  */
                collapse_table[parent->left] = 1;
                collapse_table[parent->right] = 1;

                //collapsed_nodes[parent->left].nPrimitives = 0;
                //collapsed_nodes[parent->right].nPrimitives = 0;

                /* Useful when we are collapsing nodes with
                leaf children */
                collapsed_nodes[parent->left].nPrimitives = 0xFFFF;
                collapsed_nodes[parent->right].nPrimitives = 0xFFFF;
            }

            parent_index = internal_parents[parent_index];
        }
    }

    kr_zero_memory(visit_table, internal_count * sizeof(*visit_table));

    u32 primitive_offset = 0;
    for (u32 i = 0; i < primitive_count; ++i) {
        const kr_bvh_node* parent;
        u32 parent_index = leaf_parents[i];
        while (parent_index != 0xFFFFFFFF) {
            parent = &collapsed_nodes[parent_index];

            if (kr_false == visit_table[parent_index]) {
                visit_table[parent_index] = kr_true;
                break;
            }

            if (collapse_table[parent_index] == kr_true) {
                parent_index = internal_parents[parent_index];
                continue;
            }

            i32 left_count = primitive_counts[parent->left];
            i32 right_count = primitive_counts[parent->right];

            b32 collapse_left = collapse_table[parent->left];
            b32 collapse_right = collapse_table[parent->right];

            if (left_count > 1 && collapse_left) {
                u32 offset = kr_atomic_addu(&primitive_offset, left_count);
                u32* buffer = primitives + offset;
                i32 position = kr_bvh_node_collapse(nodes, &nodes[parent->left], buffer, left_count);
                primitive_counts[parent->left] = position;

                collapsed_nodes[parent->left].nPrimitives = position;
                collapsed_nodes[parent->left].primitivesOffset = offset;
            }

            if (right_count > 1 && collapse_right) {
                u32 offset = kr_atomic_addu(&primitive_offset, right_count);
                u32* buffer = primitives + offset;
                i32 position = kr_bvh_node_collapse(nodes, &nodes[parent->right], buffer, right_count);
                primitive_counts[parent->right] = position;

                collapsed_nodes[parent->right].nPrimitives = position;
                collapsed_nodes[parent->right].primitivesOffset = offset;
            }

            parent_index = internal_parents[parent_index];
        }
    }

#define KR_RECALCULATE_COST_AFTER_COLLAPSE 0
#if KR_RECALCULATE_COST_AFTER_COLLAPSE
    kr_zero_memory(visit_table, internal_count * sizeof(*visit_table));
    kr_zero_memory(costs, node_count * sizeof(*costs));

    for (u32 i = 0; i < node_count; ++i) {
        kr_bvh_node* node = &collapsed_nodes[i];

        if (i < internal_count) {
            if (node->nPrimitives == 0) continue;

            costs[i] = ct * node->nPrimitives * kr_aabb_surface_area3(node->bounds);
        }
        else {
            if (kr_true == collapse_table[i]) continue;

            costs[i] = ct * kr_aabb_surface_area3(node->bounds);
        }
    }

    for (u32 i = 0; i < node_count; ++i) {
        kr_bvh_node* node = &collapsed_nodes[i];

        if (i < internal_count) {
            if (node->nPrimitives == 0) continue;
        }
        else {
            if (kr_true == collapse_table[i]) continue;
        }

        kr_bvh_node* parent;
        kr_u32 parent_index = parents[i];
        do {
            parent = &collapsed_nodes[parent_index];

            if (kr_false == visit_table[parent_index]) {
                visit_table[parent_index] = kr_true;
                break;
            }

            kr_scalar cost_left = costs[parent->left];
            kr_scalar cost_right = costs[parent->right];

            costs[parent_index] = ci * kr_aabb_surface_area3(parent->bounds) + cost_right + cost_left;

            parent_index = parents[parent_index];
        } while (parent_index != 0xFFFFFFFF);
    }

#endif

    kr_free((void**)&visit_table);
    kr_free((void**)&collapse_table);

    return kr_success;
}

kr_scalar
kr_bvh_sah(const kr_bvh_node* nodes, kr_i32 root_idx, kr_scalar ci, kr_scalar ct, const kr_obb3* obbs, const char* outputFile) {

    std::stack<std::pair<const kr_bvh_node*, kr_i32>> tree_stack;
    bool hasOBBs = obbs;
    const kr_bvh_node* root = &nodes[root_idx];
    const kr_obb3* root_obb = hasOBBs ? &obbs[root_idx] : nullptr;
    kr_scalar sah = 0.0f;
    u32 max_prims = 0;

    kr_scalar root_area = hasOBBs ?
        kr_obb_surface_area3(*root_obb) :
        kr_aabb_surface_area3(root->bounds);

    if (root->nPrimitives > 0) { sah += ct * root_area * root->nPrimitives; }
    else { tree_stack.push({ root, root_idx }); }

    while (!tree_stack.empty())
    {
        const kr_bvh_node* node = tree_stack.top().first;
        kr_i32 idx = tree_stack.top().second;
        tree_stack.pop();

        kr_scalar area = hasOBBs ?
            kr_obb_surface_area3(obbs[idx]) :
            kr_aabb_surface_area3(node->bounds);

        if (node->nPrimitives > 0)
        {
            if (node->nPrimitives == 0xFFFF) /* collapsed node */
                continue;
            max_prims = kr_maxu(max_prims, node->nPrimitives);
            if (node->nPrimitives == 1) {
                area = kr_aabb_surface_area3(node->bounds);
            }
            sah += ct * node->nPrimitives * area;
        }
        else
        {
            sah += ci * area;
            tree_stack.push({ &nodes[node->left], node->left });
            tree_stack.push({ &nodes[node->right], node->right });
        }
    }

    kr_scalar tree_quality = sah / root_area;

    if (outputFile)
    {
        FILE* out = fopen(outputFile, "a");
        fprintf(out, "%f,", tree_quality);
        fclose(out);
    }
    return tree_quality;
#if 0
    std::list<const kr_bvh_node*> queue;
    queue.push_back(root);

    kr_scalar sah = 0.0f;
    kr_scalar root_area = kr_aabb_surface_area3(root->bounds);

    if (root->nPrimitives > 0) {
        sah += ct * (kr_scalar)root->nPrimitives;
    }
    else {
        sah += ci;
    }

    while (!queue.empty()) {
        const kr_bvh_node* node = queue.front();
        queue.pop_front();

        if (node->nPrimitives > 0) {
            continue;
        }

        const kr_bvh_node* l = &nodes[node->left];
        const kr_bvh_node* r = &nodes[node->right];

        if (l->nPrimitives > 0) {
            sah += ct * (kr_scalar)l->nPrimitives * (kr_aabb_surface_area3(l->bounds) / root_area);
        }
        else {
            sah += ci * (kr_aabb_surface_area3(l->bounds) / root_area);
            queue.push_back(l);
        }
        if (r->nPrimitives > 0) {
            sah += ct * (kr_scalar)r->nPrimitives * (kr_aabb_surface_area3(r->bounds) / root_area);
        }
        else {
            sah += ci * (kr_aabb_surface_area3(r->bounds) / root_area);
            queue.push_back(r);
        }
    }

    return sah;
#endif
}

kr_internal std::vector<const kr_bvh_node*>
kr_bvh_node_leaves(const kr_bvh_node* nodes, const kr_bvh_node* root) {
    std::vector<const kr_bvh_node*> leaves;
    std::list<const kr_bvh_node*> queue;
    queue.push_back(root);

    while (!queue.empty()) {
        const kr_bvh_node* node = queue.front();
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
    u32 primitive_threshold) {

    KR_ALLOC_DECLARE(b32, visit_table, internal_count);

    kr_bvh_node* leaf_nodes = nodes + internal_count;
    kr_bvh_node* internal_nodes = nodes;

    kr_mat4* leaf_transforms = transforms + internal_count;
    kr_mat4* node_transforms = transforms;

    const u32* leaf_parents = parents + internal_count;
    const u32* internal_parents = parents;

    const u32* counts = primitive_counts;
    const u32* leaf_counts = counts + internal_count;
    const u32* internal_counts = counts;

    std::vector<vec3> points;
    points.resize(primitive_count * 3);

    for (u32 i = 0; i < primitive_count; ++i) {
        kr_bvh_node* leaf = &leaf_nodes[i];

        u32 primitive_id = (leaf->nPrimitives == 1) ? leaf->primitivesOffset : primitives[leaf->primitivesOffset + i];

        cuvec4 face = faces[primitive_id];
        points[0] = vertices[face.x];
        points[1] = vertices[face.y];
        points[2] = vertices[face.z];

        kr_obb3 obb = kr_points_obb(points.data(), 3, nullptr);
        obb.ext = kr_vmax3(obb.ext, kr_vof3(0.001f));
        kr_mat4 r = kr_mobb3(obb);
        kr_mat4 s = kr_mscale4(KR_INITIALIZER_CAST(vec3) { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f });
        kr_mat4 t = kr_mtranslate4(obb.mid);
        kr_mat4 trs = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));
        obbs[internal_count + i] = obb;

#ifdef KR_DITO_QUANTIZE
        trs.cols[0] = { obb.v0.x, obb.v0.y, obb.v0.z, obb.v1.x };
        trs.cols[1] = { obb.mid.x, obb.mid.y, obb.mid.z, 0 };
        trs.cols[2] = { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f, obb.ext.x * 2.0f };
#endif       
        obbs[internal_count + i] = obb;
        leaf_transforms[i] = trs;
        leaf->axis = 1;

        kr_bvh_node* parent;
        kr_u32 parent_index = leaf_parents[i];
        while (parent_index != 0xFFFFFFFF) {
            parent = &internal_nodes[parent_index];

            if (kr_false == kr_atomic_cmp_exch((kr_u32*)&visit_table[parent_index], kr_true, kr_false)) {
                break;
            }

            aabb3 aabb = parent->bounds;
            u32   count = counts[parent_index];
            if (count > primitive_threshold) {
                parent->axis = 0;
                break;
            }
        
            const auto& leaves = kr_bvh_node_leaves(nodes, parent);
            i32 vertex_count = 0;
            for (const auto& leaf : leaves) {
                u32 primitive_id = (leaf->nPrimitives == 1) ? leaf->primitivesOffset : primitives[leaf->primitivesOffset + i];

                cuvec4 face = faces[primitive_id];
                points[vertex_count + 0] = vertices[face.x];
                points[vertex_count + 1] = vertices[face.y];
                points[vertex_count + 2] = vertices[face.z];
                vertex_count += 3;
            }

            kr_obb3 obb = kr_points_obb(points.data(), vertex_count, nullptr);
            obb.ext = kr_vmax3(obb.ext, kr_vof3(0.001f));
            kr_mat4 r = kr_mobb3(obb);
            kr_mat4 s = kr_mscale4(KR_INITIALIZER_CAST(vec3) { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f });
            kr_mat4 t = kr_mtranslate4(obb.mid);
            kr_mat4 trs = kr_minverse4(kr_mmul4(t, kr_mmul4(r, s)));
#ifdef KR_DITO_QUANTIZE
            trs.cols[0] = { obb.v0.x, obb.v0.y, obb.v0.z, obb.v1.x };
            trs.cols[1] = { obb.mid.x, obb.mid.y, obb.mid.z, 0 };
            trs.cols[2] = { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f, obb.ext.x };
#endif
            obbs[parent_index] = obb;
            transforms[parent_index] = trs;

            parent->axis = (kr_aabb_surface_area3(aabb) > obb_cost * kr_obb_surface_area3(obb)) ? 1 : 0;
            parent_index = internal_parents[parent_index];
        }
    }

    kr_free((void**)&visit_table);

    return kr_success;
}