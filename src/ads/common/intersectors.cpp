#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "intersectors.h"

#include <stdio.h>

#if KR_TRAVERSAL_METRICS
i32 g_prim_intersections = 0;
i32 g_node_traversals = 0;
#endif

kr_error kr_bvh_intersect_ray(
    const kr_bvh_node* bvh, 
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    const u32* primitives,
    const kr_ray* ray,
    kr_intersection* isect) {
    if (ray->tmax < 0.0) return kr_success;
    kr_scalar min_distance = ray->tmax;
    isect->primitive = kr_invalid_index;

    cvec3 origin = ray->origin;
    cvec3 direction = ray->direction;
    cvec3 idirection = kr_vinverse3(ray->direction);

    constexpr auto KR_LBVH_STACK_SIZE = 64;
    i32 toVisitOffset = 0, currentNodeIndex = 0;
    i32 nodesToVisit[KR_LBVH_STACK_SIZE] = { 0 };
#if KR_PER_RAY_TRAVERSAL_METRICS
    i32 internal_count = 0;
    i32 leaf_count = 0;
#endif
    while (kr_true) {
        const kr_bvh_node* node = &bvh[currentNodeIndex];

        if (node->nPrimitives > 0) {
#if KR_TRAVERSAL_METRICS
            g_prim_intersections += node->nPrimitives;
#endif

#if KR_PER_RAY_TRAVERSAL_METRICS
            leaf_count += node->nPrimitives;
#endif
            for (int i = 0; i < node->nPrimitives; ++i) {
                u32 primitive_id = (node->nPrimitives == 1) ? node->primitivesOffset : primitives[node->primitivesOffset + i];

                cuvec4 face = faces[primitive_id];
                cvec3  va = vertices[face.x];
                cvec3  vb = vertices[face.y];
                cvec3  vc = vertices[face.z];

                cvec3 barys = kr_ray_triangle_intersect(direction, origin, va, vb, vc, min_distance);
                if (barys.z < min_distance) {
                    min_distance = barys.z;
                    isect->primitive = primitive_id;
                    isect->barys = KR_INITIALIZER_CAST(vec2) { barys.x, barys.y };
                }
            }
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
        else {
#if KR_TRAVERSAL_METRICS
            g_node_traversals++;
#endif
#if KR_PER_RAY_TRAVERSAL_METRICS
            internal_count++;
#endif
            const kr_bvh_node* l = &bvh[node->left];
            const kr_bvh_node* r = &bvh[node->right];

            cvec2 lisects = kr_ray_aabb_intersect(direction, origin, l->bounds, min_distance);
            cvec2 risects = kr_ray_aabb_intersect(direction, origin, r->bounds, min_distance);

            const b32 traverseChild0 = (lisects.y >= lisects.x) && (lisects.y >= 0.0) && (lisects.x <= min_distance);
            const b32 traverseChild1 = (risects.y >= risects.x) && (risects.y >= 0.0) && (risects.x <= min_distance);
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

    if (kr_invalid_index == isect->primitive)
        return kr_success;

    cuvec4 face = faces[isect->primitive];
    cvec3  va = vertices[face.x];
    cvec3  vb = vertices[face.y];
    cvec3  vc = vertices[face.z];

#if KR_PER_RAY_TRAVERSAL_METRICS
    isect->instance = (leaf_count << 16) | internal_count;
#else
    isect->instance = 0;
#endif
    isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));

    return kr_success;
}

kr_error kr_bvh_intersect(
    const kr_bvh_node* bvh, 
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    const u32* primitives,
    const kr_ray* rays,
    kr_intersection* isects, u32 ray_count) {

#if KR_TRAVERSAL_METRICS
    g_prim_intersections = g_node_traversals = 0;
#endif

    for (kr_size ray_index = 0; ray_index < ray_count; ray_index++) {
        const kr_ray* ray = &rays[ray_index];
        kr_intersection* isect = &isects[ray_index];
        kr_bvh_intersect_ray(bvh, vertices, faces, primitives, ray, isect);
    }

#if KR_TRAVERSAL_METRICS
    printf("AABB - Node Traversals %d - Primitive intersections %d\n", g_node_traversals, g_prim_intersections);
#if 0
    FILE* out = fopen("benchmark_tmp.csv", "a");
    fprintf(out, "%d, %d, %f, %f\n", g_node_traversals, g_prim_intersections, g_node_traversals / float(ray_count), g_prim_intersections / float(ray_count));
    fclose(out);
#endif
#endif

    return kr_success;
}

kr_error kr_obvh_intersect_ray(
    const kr_bvh_node* bvh,
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    const u32* primitives,
    const kr_mat43* transforms,
    const kr_ray* ray,
    kr_intersection* isect) {
    if (ray->tmax < 0.0) return kr_success;
    kr_scalar min_distance = ray->tmax;
    isect->primitive = kr_invalid_index;

    cvec3 origin = ray->origin;
    cvec3 direction = ray->direction;
    cvec3 idirection = kr_vinverse3(ray->direction);

    constexpr auto KR_LBVH_STACK_SIZE = 64;
    i32 toVisitOffset = 0, currentNodeIndex = 0;
    i32 nodesToVisit[KR_LBVH_STACK_SIZE] = { 0 };
#if KR_PER_RAY_TRAVERSAL_METRICS
    i32 internal_count = 0;
    i32 leaf_count = 0;
#endif
    while (kr_true) {
        const kr_bvh_node* node = &bvh[currentNodeIndex];

        if (node->nPrimitives > 0) {
#if KR_TRAVERSAL_METRICS
            g_prim_intersections += node->nPrimitives;
#endif
#if KR_PER_RAY_TRAVERSAL_METRICS
            leaf_count += node->nPrimitives;
#endif
            for (int i = 0; i < node->nPrimitives; ++i) {
                u32 primitive_id = (node->nPrimitives == 1) ? node->primitivesOffset : primitives[node->primitivesOffset + i];

                cuvec4 face = faces[primitive_id];
                cvec3  va = vertices[face.x];
                cvec3  vb = vertices[face.y];
                cvec3  vc = vertices[face.z];

                cvec3 barys = kr_ray_triangle_intersect(direction, origin, va, vb, vc, min_distance);
                if (barys.z < min_distance) {
                    min_distance = barys.z;
                    isect->primitive = primitive_id;
                    isect->barys = KR_INITIALIZER_CAST(vec2) { barys.x, barys.y };
                }
            }
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
        else {
#if KR_TRAVERSAL_METRICS
            g_node_traversals++;
#endif
#if KR_PER_RAY_TRAVERSAL_METRICS
            internal_count++;
#endif
            const kr_bvh_node* l = &bvh[node->left];
            const kr_bvh_node* r = &bvh[node->right];
            vec2 lisects = { 0 };
            vec2 risects = { 0 };

            if (l->axis == 1) {
                const kr_mat43* ltransform = &transforms[node->left];
                cvec3 lorigin_lcs = kr_v43transform3p(ltransform, &origin);
                cvec3 ldirection_lcs = kr_n43transform3p(ltransform, &direction);
                lisects = kr_ray_unit_aabb_intersect(ldirection_lcs, lorigin_lcs, min_distance);
            }
            else {
                lisects = kr_ray_aabb_intersect(direction, origin, l->bounds, min_distance);
            }

            if (r->axis == 1) {
                const kr_mat43* rtransform = &transforms[node->right];
                cvec3 rorigin_lcs = kr_v43transform3p(rtransform, &origin);
                cvec3 rdirection_lcs = kr_n43transform3p(rtransform, &direction);
                risects = kr_ray_unit_aabb_intersect(rdirection_lcs, rorigin_lcs, min_distance);
            }
            else {
                risects = kr_ray_aabb_intersect(direction, origin, r->bounds, min_distance);
            }

            const b32 traverseChild0 = (lisects.y >= lisects.x) && (lisects.y >= 0.0) && (lisects.x <= min_distance);
            const b32 traverseChild1 = (risects.y >= risects.x) && (risects.y >= 0.0) && (risects.x <= min_distance);
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

    if (kr_invalid_index == isect->primitive)
        return kr_success;

    cuvec4 face = faces[isect->primitive];
    cvec3  va = vertices[face.x];
    cvec3  vb = vertices[face.y];
    cvec3  vc = vertices[face.z];


#if KR_PER_RAY_TRAVERSAL_METRICS
    isect->instance = (leaf_count << 16) | internal_count;
#else
    isect->instance = 0;
#endif
    isect->geom_normal = kr_vnormalize3(kr_vcross3((kr_vsub3(vb, va)), kr_vsub3(vc, va)));

    return kr_success;
}

kr_error kr_obvh_intersect(
    const kr_bvh_node* bvh,
    const kr_vec3* vertices,
    const kr_uvec4* faces,
    const u32* primitives,
    const kr_mat43* transforms,
    const kr_ray* rays,
    kr_intersection* isects, u32 ray_count) {

#if KR_TRAVERSAL_METRICS
    g_prim_intersections = g_node_traversals = 0;
#endif

    for (kr_size ray_index = 0; ray_index < ray_count; ray_index++) {
        const kr_ray* ray = &rays[ray_index];
        kr_intersection* isect = &isects[ray_index];
        kr_obvh_intersect_ray(bvh, vertices, faces, primitives, transforms, ray, isect);
    }

#if KR_TRAVERSAL_METRICS
    printf("OBB - Node Traversals %d - Primitive intersections %d\n", g_node_traversals, g_prim_intersections);
#if 0
    FILE* out = fopen("benchmark_tmp.csv", "a");
    fprintf(out, "%d, %d, %f, %f\n", g_node_traversals, g_prim_intersections , g_node_traversals / float(ray_count), g_prim_intersections / float(ray_count));
    fclose(out);
#endif
#endif

    return kr_success;
}


kr_error kr_bvh_intersect_bounds_ray(
    const kr_bvh_node* bvh,
    const kr_ray* ray,
    kr_intersection* isect,
    i32 max_level) {

    constexpr auto KR_UNIT_CUBE = kr_aabb3{ -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    constexpr auto KR_LBVH_STACK_SIZE = 64;
 
    kr_scalar min_distance = ray->tmax;
    isect->primitive = kr_invalid_index;

    vec3 origin = ray->origin;
    vec3 direction = ray->direction;
    vec3 idirection = kr_vinverse3(ray->direction);
    vec3 normal = { 0, 0, 0 };

    i32 toVisitOffset = 0, currentNodeIndex = 0, currentNodeLevel = 0;
    i32 nodesToVisit[KR_LBVH_STACK_SIZE] = { 0 };
    i32 nodeLevels[KR_LBVH_STACK_SIZE] = { 0 };
    i32 maxLevel = (max_level < 0) ? 10000 : max_level;

    while (kr_true) {
        const kr_bvh_node* node = &bvh[currentNodeIndex];       
        cvec2 isects = kr_ray_aabb_intersect_n(direction, origin, node->bounds, min_distance, &normal);

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
            const kr_bvh_node* l = &bvh[node->left];
            const kr_bvh_node* r = &bvh[node->right];
           
            cvec2 lisects = kr_ray_aabb_intersect(direction, origin, l->bounds, min_distance);
            cvec2 risects = kr_ray_aabb_intersect(direction, origin, r->bounds, min_distance);

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

    return kr_success;
}

kr_error kr_bvh_intersect_bounds(
    const kr_bvh_node* bvh,
    const kr_ray* rays,
    kr_intersection* isects, u32 ray_count, i32 max_level) {

    for (kr_size ray_index = 0; ray_index < ray_count; ray_index++) {
        const kr_ray* ray = &rays[ray_index];
        kr_intersection* isect = &isects[ray_index];
        kr_bvh_intersect_bounds_ray(bvh, ray, isect, max_level);
    }

    return kr_success;
}

kr_error kr_obvh_intersect_bounds_ray(
    const kr_bvh_node* bvh,
    const kr_mat4* transforms,
    const kr_ray* ray,
    kr_intersection* isect,
    i32 max_level) {

    constexpr auto KR_LBVH_STACK_SIZE = 64;

    kr_scalar min_distance = ray->tmax;
    isect->primitive = kr_invalid_index;

    vec3 origin = ray->origin;
    vec3 direction = ray->direction;
    vec3 idirection = kr_vinverse3(ray->direction);
    vec3 normal = { 0, 0, 0 };

    i32 toVisitOffset = 0, currentNodeIndex = 0, currentNodeLevel = 0;
    i32 nodesToVisit[KR_LBVH_STACK_SIZE] = { 0 };
    i32 nodeLevels[KR_LBVH_STACK_SIZE] = { 0 };
    i32 maxLevel = (max_level < 0) ? 10000 : max_level;

    while (kr_true) {
        const kr_bvh_node* node = &bvh[currentNodeIndex];
        vec2 isects = { 0 };
        if (node->axis == 1) {
            cmat4* transform = &transforms[currentNodeIndex];
            cvec3 origin_lcs = kr_vtransform3p(transform, &origin);
            cvec3 direction_lcs = kr_ntransform3p(transform, &direction);
            isects = kr_ray_unit_aabb_intersect_n(direction_lcs, origin_lcs, min_distance, &normal);
            normal = kr_vnormalize3(kr_ntransform3(kr_minverse4(*transform), normal));
        }
        else {
            isects = kr_ray_aabb_intersect_n(direction, origin, node->bounds, min_distance, &normal);
        }

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
            const kr_bvh_node* l = &bvh[node->left];
            const kr_bvh_node* r = &bvh[node->right];

            vec2 lisects = { 0 };
            vec2 risects = { 0 };

            if (l->axis == 1) {
                const kr_mat4* ltransform = &transforms[node->left];
                cvec3 lorigin_lcs = kr_vtransform3p(ltransform, &origin);
                cvec3 ldirection_lcs = kr_ntransform3p(ltransform, &direction);
                lisects = kr_ray_unit_aabb_intersect_n(ldirection_lcs, lorigin_lcs, min_distance, &normal);
            }
            else {
                lisects = kr_ray_aabb_intersect_n(direction, origin, l->bounds, min_distance, &normal);
            }

            if (r->axis == 1) {
                const kr_mat4* rtransform = &transforms[node->right];
                cvec3 rorigin_lcs = kr_vtransform3p(rtransform, &origin);
                cvec3 rdirection_lcs = kr_ntransform3p(rtransform, &direction);
                risects = kr_ray_unit_aabb_intersect_n(rdirection_lcs, rorigin_lcs, min_distance, &normal);
            }
            else {
                risects = kr_ray_aabb_intersect_n(direction, origin, r->bounds, min_distance, &normal);
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

    return kr_success;
}

kr_error kr_obvh_intersect_bounds(
    const kr_bvh_node* bvh,
    const kr_mat4* transforms,
    const kr_ray* rays,
    kr_intersection* isects, u32 ray_count, i32 max_level) {

    for (kr_size ray_index = 0; ray_index < ray_count; ray_index++) {
        const kr_ray* ray = &rays[ray_index];
        kr_intersection* isect = &isects[ray_index];
        kr_obvh_intersect_bounds_ray(bvh, transforms, ray, isect, max_level);
    }

    return kr_success;
}