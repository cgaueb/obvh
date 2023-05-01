#define KR_VECMATH_IMPL
#include "vecmath.h"

#include "queue.h"
#include "geometry.h"
#include "scene.h"
#include "logger.h"

#include "algorithm/dito/dito.h"

#include <cstring>
#include <cstdio>

#include <vector>
#include <array>
#include <list>
#include <unordered_map>
#include <chrono>
#include <algorithm>

unsigned int Hash(float f)
{
	unsigned int ui;
	memcpy(&ui, &f, sizeof(float));
	//return ui & 0xfffff000;
	return ui & 0xfffffff8;
}

struct KeyFuncs
{
	size_t operator()(const vec3& k)const
	{
		return Hash(k.x) ^ Hash(k.y) ^ Hash(k.z);
	}

	bool operator()(const vec3& a, const vec3& b)const
	{
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};

struct CachedTriangle {
	vec3 a, b, c;
};

struct TriKeyFuncs {
	size_t operator()(const CachedTriangle& k) const {
		return Hash(k.a.x) ^ Hash(k.a.y) ^ Hash(k.a.z) ^
			Hash(k.b.x) ^ Hash(k.b.y) ^ Hash(k.b.z) ^
			Hash(k.b.x) ^ Hash(k.b.y) ^ Hash(k.b.z);
	}

	bool operator()(const CachedTriangle& a, const CachedTriangle& b) const {
		return (
			(a.a.x == b.a.x && a.a.y == b.a.y && a.a.z == b.a.z) &&
			(a.b.x == b.b.x && a.b.y == b.b.y && a.b.z == b.b.z) &&
			(a.c.x == b.c.x && a.c.y == b.c.y && a.c.z == b.c.z)
		);
	}
};

typedef std::unordered_map<vec3, int, KeyFuncs, KeyFuncs> VertexCache;
typedef std::unordered_map<CachedTriangle, int, TriKeyFuncs, TriKeyFuncs> TriCache;

kr_error
kr_scene_deduplicate(kr_scene* scene) {
	kr_size object_count = scene->object_count;
	kr_size instance_count = scene->instance_count;
	u32 primitive_count = 0;
	u32 attr_count = 0;
	TriCache cache;

	kr_size mesh_index = 0;
	kr_size mesh_count = kr_queue_size(scene->objects);

	kr_object_instance* instances;
	kr_object* objects;

	kr_queue_init(instances, 1);
	kr_queue_init(objects, 1);

	for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
		kr_object_instance* instance = &scene->instances[instance_index];
		kr_object* object = &scene->objects[instance->object_id];
		kr_object cleaned_object = { };

		switch (object->type) {
		case KR_OBJECT_AABB:
			break;
		case KR_OBJECT_MESH: {
			for (u32 i = 0; i < object->as_mesh.face_count; ++i) {
				CachedTriangle tri = {
					object->as_mesh.vertices[object->as_mesh.faces[i].x],
					object->as_mesh.vertices[object->as_mesh.faces[i].y],
					object->as_mesh.vertices[object->as_mesh.faces[i].z]
				};
				aabb3 box = kr_aabb_empty3();
				box = kr_aabb_expand3(box, tri.a);
				box = kr_aabb_expand3(box, tri.b);
				box = kr_aabb_expand3(box, tri.c);
				
				if (kr_aabb_surface_area3(box) < 0.00000001) {
					vec3 ext = kr_aabb_extents3(box);
					float sah = kr_aabb_surface_area3(box);
					//printf("Ludacris\n");
					continue;
				}

				if (std::end(cache) != cache.find(tri)) {
					int face_idx = cache[tri];
					CachedTriangle tri2 = {
						object->as_mesh.vertices[object->as_mesh.faces[face_idx].x],
						object->as_mesh.vertices[object->as_mesh.faces[face_idx].y],
						object->as_mesh.vertices[object->as_mesh.faces[face_idx].z]
					};
				}
				cache[tri] = i;
			}


			cleaned_object.type = KR_OBJECT_MESH;
			cleaned_object.aabb = kr_aabb_empty3();
			cleaned_object.as_mesh.face_count = (kr_size)cache.size();
			cleaned_object.as_mesh.attr_count = (kr_size)cache.size() * 3;

			cleaned_object.as_mesh.faces = (kr_uvec4*)kr_aligned_allocate(cleaned_object.as_mesh.face_count * sizeof(*cleaned_object.as_mesh.faces), kr_align_of(kr_uvec4));
			cleaned_object.as_mesh.vertices = (kr_vec3*)kr_aligned_allocate(cleaned_object.as_mesh.attr_count * sizeof(*cleaned_object.as_mesh.vertices), kr_align_of(kr_vec3));
			cleaned_object.as_mesh.normals = (kr_vec3*)kr_aligned_allocate(cleaned_object.as_mesh.attr_count * sizeof(*cleaned_object.as_mesh.normals), kr_align_of(kr_vec3));
			cleaned_object.as_mesh.uvs = (kr_vec2*)kr_aligned_allocate(cleaned_object.as_mesh.attr_count * sizeof(*cleaned_object.as_mesh.uvs), kr_align_of(kr_vec2));

			//kr_queue_reserve(cleaned_object.as_mesh.faces, cleaned_object.as_mesh.face_count);
			//kr_queue_reserve(cleaned_object.as_mesh.vertices, cleaned_object.as_mesh.attr_count);
			//kr_queue_reserve(cleaned_object.as_mesh.normals, cleaned_object.as_mesh.attr_count);
			//kr_queue_reserve(cleaned_object.as_mesh.uvs, cleaned_object.as_mesh.attr_count);

			kr_size face_index = 0;
			kr_size attr_index = 0;
			for (const auto& p : cache) {
				const auto& tri = p.first;
				const auto& tri_idx = p.second;

				cleaned_object.as_mesh.vertices[attr_index] = object->as_mesh.vertices[object->as_mesh.faces[tri_idx].x];
				cleaned_object.as_mesh.normals[attr_index] = object->as_mesh.normals[object->as_mesh.faces[tri_idx].x];
				cleaned_object.as_mesh.uvs[attr_index] = object->as_mesh.uvs[object->as_mesh.faces[tri_idx].x];
				cleaned_object.aabb = kr_aabb_expand3(cleaned_object.aabb, cleaned_object.as_mesh.vertices[attr_index]);
				attr_index++;
				
				cleaned_object.as_mesh.vertices[attr_index] = object->as_mesh.vertices[object->as_mesh.faces[tri_idx].y];
				cleaned_object.as_mesh.normals[attr_index] = object->as_mesh.normals[object->as_mesh.faces[tri_idx].y];
				cleaned_object.as_mesh.uvs[attr_index] = object->as_mesh.uvs[object->as_mesh.faces[tri_idx].y];
				cleaned_object.aabb = kr_aabb_expand3(cleaned_object.aabb, cleaned_object.as_mesh.vertices[attr_index]);
				attr_index++;

				cleaned_object.as_mesh.vertices[attr_index] = object->as_mesh.vertices[object->as_mesh.faces[tri_idx].z];
				cleaned_object.as_mesh.normals[attr_index] = object->as_mesh.normals[object->as_mesh.faces[tri_idx].z];
				cleaned_object.as_mesh.uvs[attr_index] = object->as_mesh.uvs[object->as_mesh.faces[tri_idx].z];
				cleaned_object.aabb = kr_aabb_expand3(cleaned_object.aabb, cleaned_object.as_mesh.vertices[attr_index]);
				attr_index++;

				cleaned_object.as_mesh.faces[face_index] = { (u32)attr_index - 3, (u32)attr_index - 2, (u32)attr_index - 1, 0 };

				face_index++;
			}

			kr_object_instance cleaned_instance = *instance;

			kr_queue_push(instances, cleaned_instance);
			kr_queue_push(objects, cleaned_object);

		} break;
		default:
			break;
		}
	}

	for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		kr_object* object = &scene->objects[mesh_index];

		kr_aligned_free((void**)&object->as_mesh.faces);
		kr_aligned_free((void**)&object->as_mesh.vertices);
		kr_aligned_free((void**)&object->as_mesh.normals);
		kr_aligned_free((void**)&object->as_mesh.uvs);
	}
    
	kr_queue_release(scene->objects);
	kr_queue_release(scene->instances);

	scene->objects = objects;
	scene->instances = instances;

	scene->instance_count = kr_queue_size(scene->instances);
	scene->object_count = kr_queue_size(scene->objects);

    kr_log("Scene (Deduplicated) face count: %d attr_count: %d\n", scene->objects[0].as_mesh.face_count, scene->objects[0].as_mesh.attr_count);
    kr_log("Scene (Deduplicated) AABB [%f %f %f] x [%f %f %f]\n", scene->aabb.min.x, scene->aabb.min.y, scene->aabb.min.z, scene->aabb.max.x, scene->aabb.max.y, scene->aabb.max.z);

	return kr_success;
}

kr_obb3
kr_triangle_obb(kr_vec3 v0, kr_vec3 v1, kr_vec3 v2) {
	kr_obb3 obb_out = { 0 };

    kr_aabb3 aabb = kr_aabb_create3(v0, v0);
    aabb = kr_aabb_expand3(aabb, v1);
    aabb = kr_aabb_expand3(aabb, v2);

    cvec3 diagonal = kr_aabb_extents3(aabb);

    //cvec3 right = kr_vnormalize3

    printf("Hello\n");

#if 0	
    cvec3 e0 = kr_vsub3(v1, v0);
	cvec3 e1 = kr_vsub3(v2, v0);
	cvec3 e2 = kr_vsub3(v2, v1);
	
	cvec3 axis_z = kr_vnormalize3(kr_vcross3(e0, e1));
	cvec3 axis_y = kr_vnormalize3(e0);
	cvec3 axis_x = kr_vcross3(axis_y, axis_z);

	kr_obb3 obb;
	obb.v0 = axis_x;
	obb.v1 = axis_y;
	obb.v2 = axis_x;

#endif
	return obb_out;
}

kr_obb3
kr_points_obb(const kr_vec3* points, kr_size count, kr_scalar* elapsed_time) {
	DiTO::OBB<float> obb;

	auto start = std::chrono::high_resolution_clock::now();
	DiTO::DiTO_14<float>((DiTO::Vector<float>*) points, (int)count, obb);
	auto stop = std::chrono::high_resolution_clock::now();
	if (elapsed_time) *elapsed_time = std::chrono::duration<float, std::milli>(stop - start).count();
	//if (count == 3) return kr_triangle_obb(points[0], points[1], points[2]);

	kr_obb3 obb_out = { 0 };

	obb_out.v0 = { obb.v0.x, obb.v0.y, obb.v0.z };
	obb_out.v1 = { obb.v1.x, obb.v1.y, obb.v1.z };
	obb_out.v2 = { obb.v2.x, obb.v2.y, obb.v2.z };
	obb_out.mid = { obb.mid.x, obb.mid.y, obb.mid.z };
	obb_out.ext = { obb.ext.x, obb.ext.y, obb.ext.z };
	obb_out.ext = kr_vmax3(obb_out.ext, kr_vof3(0.001f));

	return obb_out;
}

constexpr auto g_cube_faces = std::array<uvec3, 12> { {
    {0, 1, 2}, { 0, 2, 3 }, { 4, 5, 6 },
    { 4, 6, 7 }, { 8, 9, 10 }, { 8, 10, 11 },
    { 12, 13, 14 }, { 12, 14, 15 }, { 16, 17, 18 },
    { 16, 18, 19 }, { 20, 21, 22 }, { 20, 22, 23 },
} };

constexpr auto g_cube_normals = std::array<vec3, 24> { {
    {0.000000, 1.000000, 0.000000},
    { 0.000000, 1.000000, 0.000000 },
    { 0.000000, 1.000000, 0.000000 },
    { 0.000000, 1.000000, 0.000000 },
    { 0.000000, 0.000000, 1.000000 },
    { 0.000000, 0.000000, 1.000000 },
    { 0.000000, 0.000000, 1.000000 },
    { 0.000000, 0.000000, 1.000000 },
    { -1.000000, 0.000000, 0.000000 },
    { -1.000000, 0.000000, 0.000000 },
    { -1.000000, 0.000000, 0.000000 },
    { -1.000000, 0.000000, 0.000000 },
    { 0.000000, -1.000000, 0.000000 },
    { 0.000000, -1.000000, 0.000000 },
    { 0.000000, -1.000000, 0.000000 },
    { 0.000000, -1.000000, 0.000000 },
    { 1.000000, 0.000000, 0.000000 },
    { 1.000000, 0.000000, 0.000000 },
    { 1.000000, 0.000000, 0.000000 },
    { 1.000000, 0.000000, 0.000000 },
    { 0.000000, 0.000000, -1.000000 },
    { 0.000000, 0.000000, -1.000000 },
    { 0.000000, 0.000000, -1.000000 },
    { 0.000000, 0.000000, -1.000000 },
    } };

constexpr auto g_cube_vertices = std::array<vec3, 24> { {
    {1.000000, 1.000000, -1.000000},
    { -1.000000, 1.000000, -1.000000 },
    { -1.000000, 1.000000, 1.000000 },
    { 1.000000, 1.000000, 1.000000 },
    { 1.000000, -1.000000, 1.000000 },
    { 1.000000, 1.000000, 1.000000 },
    { -1.000000, 1.000000, 1.000000 },
    { -1.000000, -1.000000, 1.000000 },
    { -1.000000, -1.000000, 1.000000 },
    { -1.000000, 1.000000, 1.000000 },
    { -1.000000, 1.000000, -1.000000 },
    { -1.000000, -1.000000, -1.000000 },
    { -1.000000, -1.000000, -1.000000 },
    { 1.000000, -1.000000, -1.000000 },
    { 1.000000, -1.000000, 1.000000 },
    { -1.000000, -1.000000, 1.000000 },
    { 1.000000, -1.000000, -1.000000 },
    { 1.000000, 1.000000, -1.000000 },
    { 1.000000, 1.000000, 1.000000 },
    { 1.000000, -1.000000, 1.000000 },
    { -1.000000, -1.000000, -1.000000 },
    { -1.000000, 1.000000, -1.000000 },
    { 1.000000, 1.000000, -1.000000 },
    { 1.000000, -1.000000, -1.000000 },
} };

constexpr auto g_cube_uvs = std::array<vec2, 24> { {
    {0.625000, 0.500000 },
    { 0.875000, 0.500000 },
    { 0.875000, 0.250000 },
    { 0.625000, 0.250000 },
    { 0.375000, 0.250000 },
    { 0.625000, 0.250000 },
    { 0.625000, 0.000000 },
    { 0.375000, 0.000000 },
    { 0.375000, 1.000000 },
    { 0.625000, 1.000000 },
    { 0.625000, 0.750000 },
    { 0.375000, 0.750000 },
    { 0.125000, 0.500000 },
    { 0.375000, 0.500000 },
    { 0.375000, 0.250000 },
    { 0.125000, 0.250000 },
    { 0.375000, 0.500000 },
    { 0.625000, 0.500000 },
    { 0.625000, 0.250000 },
    { 0.375000, 0.250000 },
    { 0.375000, 0.750000 },
    { 0.625000, 0.750000 },
    { 0.625000, 0.500000 },
    { 0.375000, 0.500000 },
} };

struct kr_node_with_level {
    const kr_bvh_node* node;
    i32 level;
};


kr_internal kr_error
bvh_node_print(const kr_bvh_node* nodes, const kr_bvh_node* root) {
    std::list<const kr_bvh_node*> queue;
    queue.push_back(root);

    while (!queue.empty()) {
        const kr_bvh_node* node = queue.front();
        queue.pop_front();

        ptrdiff_t offset = node - nodes;
        if (node->nPrimitives > 0) {
            printf("Leaf [%d] [%d]\n", offset, node->nPrimitives);

            continue;
        }
        printf("Node [%d] [%d]\n", offset, node->nPrimitives);
        queue.push_back(&nodes[node->left]);
        queue.push_back(&nodes[node->right]);
    }

    return kr_success;
}

kr_error
kr_bvh_node_print(const kr_bvh_node* nodes, const kr_bvh_node* root) {

    return bvh_node_print(nodes, root);
}

kr_error
kr_bvh_nodes_export(
    const kr_bvh_node* nodes,
    const kr_bvh_node* root,
    const kr_mat4* transforms,
    const kr_obb3* obbs,
    i32 depth,
    const char* filename) {
    kr_b32 seperate_object_per_node = false;

    std::array<std::vector<const kr_bvh_node*>, 256> levels = { };
    i32 current_level = 0;
    std::list<kr_node_with_level> queue;
    queue.push_back({ root, current_level});

    while (!queue.empty()) {
        const auto data = queue.front();
        queue.pop_front();

        if (data.level > 255) continue;

        levels[data.level].push_back(data.node);

        if (data.node->nPrimitives > 0) continue;

        queue.push_back({ &nodes[data.node->left], data.level + 1 });
        queue.push_back({ &nodes[data.node->right], data.level + 1 });
    }

    i32 index = 0;
    i32 prev_index = 0;
    i32 current_attr_count = 0;

    char obj_filename[128] = { 0 };
    sprintf(obj_filename, "%s.obj", filename);
    char mtl_filename[128] = { 0 };
    sprintf(mtl_filename, "%s.mtl", filename);

    FILE* obj = fopen(obj_filename, "w");
    FILE* mtl = fopen(mtl_filename, "w");

    for (index = 1; index < levels.size(); index++) {
        const auto& level = levels[index];
        prev_index = index - 1;
        if (index == 0) continue;
        if (level.empty()) break;

        for (const auto& node : levels[prev_index]) {
            if (node->nPrimitives == 0) continue;
            levels[index].push_back(node);
        }
    }


    for (auto& level : levels) {
        std::sort(level.begin(), level.end(),
            [](const kr_bvh_node* a, const kr_bvh_node* b) -> bool
            {
                return a->nPrimitives < b->nPrimitives;
            });
    }

    index = 0;
    index = 0;
    for (const auto& level : levels) {
        index++;
        if ((depth >= 0) && index != (depth + 1)) continue;
        if (level.empty()) continue;

        char name[128] = { 0 };
        sprintf(name, "obj_%d.obj", index);

        i32 node_index = 0;

        if (!seperate_object_per_node) {
            fprintf(obj, "o BVH_%d\n", index);
            fprintf(obj, "g BVH_%d\n", index);
            fprintf(obj, "\n");
            fprintf(obj, "usemtl MAT_BVH_%d\n", index);
        }
        for (const auto node : level) {
            //if (node->nPrimitives > 0) continue;
            node_index++;
            
            //if (node_index == 9032) {
            //if (node_index == 5147) {
            if (node_index == 6020) {
                ptrdiff_t offest = node - nodes;
                //printf("Here %d == 9032 685480\n", offest);
                //printf("Here %d == 5147 120300\n", offest);
                //printf("Here %d == 6020 128249\n", offest);
                //kr_bvh_node_print(nodes, node);
            }
            std::array<uvec3, 12> faces = g_cube_faces;
            std::array<vec3, 24>  normals = g_cube_normals;
            std::array<vec3, 24>  vertices = g_cube_vertices;
            std::array<vec2, 24>  uvs = g_cube_uvs;

            i32 attr_count = (i32)vertices.size();
            i32 face_count = (i32)faces.size();

            if (kr_null != transforms) {
                for (auto& vert : vertices) {
                    vert = kr_vmul31(vert, 0.5f);
                }
                ptrdiff_t offest = node - nodes;
                kr_mat4 transform = kr_minverse4(transforms[offest]);
                kr_obb3 obb = obbs[offest];

                kr_mat4 r = kr_mobb3(obb);
                kr_mat4 t = kr_mtranslate4(obb.mid);

                kr_vec3 x = kr_vto43(r.cols[0]);
                kr_vec3 y = kr_vto43(r.cols[1]);
                kr_vec3 z = kr_vto43(r.cols[2]);

                if (kr_vdot3(x, { 1, 0, 0 }) < 0.0f) {
                    obb.ext.x *= -1.0f;
                    x = kr_vnegate3(x);
                }
                if (kr_vdot3(y, { 0, 1, 0 }) < 0.0f) {
                    obb.ext.y *= -1.0f;
                    y = kr_vnegate3(y);
                }
                if (kr_vdot3(z, { 0, 0, 1 }) < 0.0f) {
                    obb.ext.z *= -1.0f;
                    z = kr_vnegate3(z);
                }

                r.cols[0] = { x.x, x.y, x.z, 0.0f };
                r.cols[1] = { y.x, y.y, y.z, 0.0f };
                r.cols[2] = { z.x, z.y, z.z, 0.0f };

                kr_mat4 s = kr_mscale4(KR_INITIALIZER_CAST(vec3) { obb.ext.x * 2.0f, obb.ext.y * 2.0f, obb.ext.z * 2.0f });

                transform = (kr_mmul4(t, kr_mmul4(r, s)));

                for (auto& vert : vertices) {
                    vert = kr_vtransform3p(&transform, &vert);
                }
                for (auto& normal : normals) {
                    normal = kr_ntransform3p(&transform, &normal);
                }
            }
            else {
                cvec3 center = kr_aabb_center3(node->bounds);
                cvec3 extents = kr_vmul31(kr_aabb_extents3(node->bounds), 0.5f);

                for (auto& vert : vertices) {
                    vert = kr_vmul3(vert, extents);
                }
                for (auto& vert : vertices) {
                    vert = kr_vadd3(vert, center);
                }
            }

            if (seperate_object_per_node) {
                fprintf(obj, "o BVH_%d_%d\n", index, node_index);
                fprintf(obj, "g BVH_%d_%d\n", index, node_index);
                fprintf(obj, "\n");
                fprintf(obj, "usemtl MAT_BVH_%d\n", index);
            }

            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "v %f %f %f\n", vertices[i].x, vertices[i].y, vertices[i].z);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vt %f %f\n", uvs[i].x, uvs[i].y);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vn %f %f %f\n", normals[i].x, normals[i].y, normals[i].z);
            }
            fprintf(obj, "\n");

            for (int i = 0; i < face_count; i++) {
                uvec3 relative_face = faces[i];
                uvec3 face = { relative_face.x + 1 + current_attr_count, relative_face.y + 1 + current_attr_count, relative_face.z + 1 + current_attr_count };
                fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
            }
            fprintf(obj, "\n");
            current_attr_count += int(attr_count);
            //object_index++;
        }
        if (!seperate_object_per_node) {
            fprintf(obj, "o BVH_%d_Leaves\n", index);
            fprintf(obj, "g BVH_%d_Leaves\n", index);
            fprintf(obj, "\n");
            fprintf(obj, "usemtl MAT_BVH_Leaves_%d\n", index);
        }
        for (const auto node : level) {
            continue;
            if (node->nPrimitives == 0) continue;
            node_index++;

            std::array<uvec3, 12> faces = g_cube_faces;
            std::array<vec3, 24>  normals = g_cube_normals;
            std::array<vec3, 24>  vertices = g_cube_vertices;
            std::array<vec2, 24>  uvs = g_cube_uvs;

            i32 attr_count = (i32)vertices.size();
            i32 face_count = (i32)faces.size();


            if (kr_null != transforms) {
                for (auto& vert : vertices) {
                    vert = kr_vmul31(vert, 0.5f);
                }
                ptrdiff_t offest = node - nodes;
                const kr_mat4 transform = kr_minverse4(transforms[offest]);

                for (auto& vert : vertices) {
                    vert = kr_vtransform3p(&transform, &vert);
                }
                for (auto& normal : normals) {
                    normal = kr_ntransform3p(&transform, &normal);
                }
            }
            else {
                cvec3 center = kr_aabb_center3(node->bounds);
                cvec3 extents = kr_vmul31(kr_aabb_extents3(node->bounds), 0.5f);

                for (auto& vert : vertices) {
                    vert = kr_vmul3(vert, extents);
                }
                for (auto& vert : vertices) {
                    vert = kr_vadd3(vert, center);
                }
            }

            if (seperate_object_per_node) {
                fprintf(obj, "o BVH_%d_%d\n", index, node_index);
                fprintf(obj, "g BVH_%d_%d\n", index, node_index);
                fprintf(obj, "\n");
                fprintf(obj, "usemtl MAT_BVH_Leaves_%d\n", index);
            }

            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "v %f %f %f\n", vertices[i].x, vertices[i].y, vertices[i].z);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vt %f %f\n", uvs[i].x, uvs[i].y);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vn %f %f %f\n", normals[i].x, normals[i].y, normals[i].z);
            }
            fprintf(obj, "\n");

            for (int i = 0; i < face_count; i++) {
                uvec3 relative_face = faces[i];
                uvec3 face = { relative_face.x + 1 + current_attr_count, relative_face.y + 1 + current_attr_count, relative_face.z + 1 + current_attr_count };
                fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
            }
            fprintf(obj, "\n");
            current_attr_count += int(attr_count);
        }

        fprintf(mtl, "newmtl MAT_BVH_%d\n", index);
        fprintf(mtl, "\tNs %f\n", 100.0);
        fprintf(mtl, "\tNi %f\n", 1.5);
        fprintf(mtl, "\td %f\n", 1.0);
        fprintf(mtl, "\tTr %f\n", 0.0);
        fprintf(mtl, "\tTf %f %f %f\n", 0.5, 0.5, 0.5);
        float value = 0.75f;
        fprintf(mtl, "\tKa %f %f %f\n", 0.2f, 0.2f, 0.2f);
        fprintf(mtl, "\tKd %f %f %f\n", value, value, value);
        fprintf(mtl, "\tKs %f %f %f\n", 1.0f, 1.0f, 1.0f);
        fprintf(mtl, "\tKe %f %f %f\n", 0.0f, 0.0f, 0.0f);
        fprintf(mtl, "\tillum %d\n", 2);
        fprintf(mtl, "\n");

        fprintf(mtl, "newmtl MAT_BVH_Leaves_%d\n", index);
        fprintf(mtl, "\tNs %f\n", 100.0);
        fprintf(mtl, "\tNi %f\n", 1.5);
        fprintf(mtl, "\td %f\n", 1.0);
        fprintf(mtl, "\tTr %f\n", 0.0);
        fprintf(mtl, "\tTf %f %f %f\n", 0.5, 0.5, 0.5);
        value = 0.1f;
        fprintf(mtl, "\tKa %f %f %f\n", 0.2f, 0.2f, 0.2f);
        fprintf(mtl, "\tKd %f %f %f\n", value, value, value);
        fprintf(mtl, "\tKs %f %f %f\n", 1.0f, 1.0f, 1.0f);
        fprintf(mtl, "\tKe %f %f %f\n", 0.0f, 0.0f, 0.0f);
        fprintf(mtl, "\tillum %d\n", 2);
        fprintf(mtl, "\n");
    }

    fclose(obj);
    fclose(mtl);

    return kr_success;
}
#if 0
kr_error
kr_bvh_nodes_export(const kr_bvh_node* nodes, const kr_bvh_node* root, const char* filename) {
    kr_b32 seperate_object_per_node = kr_false;

    std::array<std::vector<const kr_bvh_node*>, 256> levels = { };
    i32 current_level = 0;
    std::list<kr_node_with_level> queue;
    queue.push_back({ root, kr_null, current_level });

    while (!queue.empty()) {
        const auto data = queue.front();
        queue.pop_front();

        if (data.level > 255) continue;

        levels[data.level].push_back(data.node);

        if (data.node->nPrimitives > 0) continue;

        queue.push_back({ &nodes[data.node->left], kr_null, data.level + 1 });
        queue.push_back({ &nodes[data.node->right], kr_null, data.level + 1 });
    }


    i32 index = 0;
    i32 prev_index = 0;
    i32 current_attr_count = 0;

    char obj_filename[128] = { 0 };
    sprintf(obj_filename, "%s.obj", filename);
    char mtl_filename[128] = { 0 };
    sprintf(mtl_filename, "%s.mtl", filename);

    FILE* obj = fopen(obj_filename, "w");
    FILE* mtl = fopen(mtl_filename, "w");

    fprintf(obj, "mtllib %s\n\n", mtl_filename);

    for (index = 1; index < levels.size(); index++) {
        const auto& level = levels[index];
        prev_index = index - 1;
        if (index == 0) continue;
        if (level.empty()) break;

        for (const auto& node : levels[prev_index]) {
            if (node->nPrimitives == 0) continue;
            levels[index].push_back(node);
        }
    }

    for (auto& level : levels) {
        std::sort(level.begin(), level.end(),
        [](const kr_bvh_node* a, const kr_bvh_node* b) -> bool
        {
            return a->nPrimitives < b->nPrimitives;
        });
    }

    index = 0;
    for (const auto& level : levels) {
        index++;
        //if (index > 3) continue;
        if (level.empty()) continue;

        char name[128] = { 0 };
        sprintf(name, "obj_%d.obj", index);

        i32 node_index = 0;

        if (!seperate_object_per_node) {
            fprintf(obj, "o BVH_%d\n", index);
            fprintf(obj, "g BVH_%d\n", index);
            fprintf(obj, "\n");
            fprintf(obj, "usemtl MAT_BVH_%d\n", index);
        }
        for (const auto node : level) {
            if (node->nPrimitives > 0) continue;
            node_index++;

            std::array<uvec3, 12> faces = g_cube_faces;
            std::array<vec3, 24>  normals = g_cube_normals;
            std::array<vec3, 24>  vertices = g_cube_vertices;
            std::array<vec2, 24>  uvs = g_cube_uvs;
            
            i32 attr_count = (i32)vertices.size();
            i32 face_count = (i32)faces.size();

            cvec3 center = kr_aabb_center3(node->bounds);
            cvec3 extents = kr_vmul31(kr_aabb_extents3(node->bounds), 0.5f);

            for (auto& vert : vertices) {
                vert = kr_vmul3(vert, extents);
            }
            for (auto& vert : vertices) {
                vert = kr_vadd3(vert, center);
            }

            if (seperate_object_per_node) {
                fprintf(obj, "o BVH_%d_%d\n", index, node_index);
                fprintf(obj, "g BVH_%d_%d\n", index, node_index);
                fprintf(obj, "\n");
                fprintf(obj, "usemtl MAT_BVH_%d\n", index);
            }

            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "v %f %f %f\n", vertices[i].x, vertices[i].y, vertices[i].z);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vt %f %f\n", uvs[i].x, uvs[i].y);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vn %f %f %f\n", normals[i].x, normals[i].y, normals[i].z);
            }
            fprintf(obj, "\n");

            for (int i = 0; i < face_count; i++) {
                uvec3 relative_face = faces[i];
                uvec3 face = { relative_face.x + 1 + current_attr_count, relative_face.y + 1 + current_attr_count, relative_face.z + 1 + current_attr_count };
                fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
            }
            fprintf(obj, "\n");
            current_attr_count += int(attr_count);
            //object_index++;
        }
        if (!seperate_object_per_node) {
            fprintf(obj, "o BVH_%d_Leaves\n", index);
            fprintf(obj, "g BVH_%d_Leaves\n", index);
            fprintf(obj, "\n");
            fprintf(obj, "usemtl MAT_BVH_Leaves_%d\n", index);
        }
        for (const auto node : level) {
            if (node->nPrimitives == 0) continue;
            node_index++;

            std::array<uvec3, 12> faces = g_cube_faces;
            std::array<vec3, 24>  normals = g_cube_normals;
            std::array<vec3, 24>  vertices = g_cube_vertices;
            std::array<vec2, 24>  uvs = g_cube_uvs;

            i32 attr_count = (i32)vertices.size();
            i32 face_count = (i32)faces.size();

            cvec3 center = kr_aabb_center3(node->bounds);
            cvec3 extents = kr_vmul31(kr_aabb_extents3(node->bounds), 0.5f);

            for (auto& vert : vertices) {
                vert = kr_vmul3(vert, extents);
            }
            for (auto& vert : vertices) {
                vert = kr_vadd3(vert, center);
            }

            if (seperate_object_per_node) {
                fprintf(obj, "o BVH_%d_%d\n", index, node_index);
                fprintf(obj, "g BVH_%d_%d\n", index, node_index);
                fprintf(obj, "\n");
                fprintf(obj, "usemtl MAT_BVH_Leaves_%d\n", index);
            }

            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "v %f %f %f\n", vertices[i].x, vertices[i].y, vertices[i].z);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vt %f %f\n", uvs[i].x, uvs[i].y);
            }
            fprintf(obj, "\n");
            for (int i = 0; i < attr_count; i++) {
                fprintf(obj, "vn %f %f %f\n", normals[i].x, normals[i].y, normals[i].z);
            }
            fprintf(obj, "\n");

            for (int i = 0; i < face_count; i++) {
                uvec3 relative_face = faces[i];
                uvec3 face = { relative_face.x + 1 + current_attr_count, relative_face.y + 1 + current_attr_count, relative_face.z + 1 + current_attr_count };
                fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
            }
            fprintf(obj, "\n");
            current_attr_count += int(attr_count);
        }

        fprintf(mtl, "newmtl MAT_BVH_%d\n", index);
        fprintf(mtl, "\tNs %f\n", 100.0);
        fprintf(mtl, "\tNi %f\n", 1.5);
        fprintf(mtl, "\td %f\n", 1.0);
        fprintf(mtl, "\tTr %f\n", 0.0);
        fprintf(mtl, "\tTf %f %f %f\n", 0.5, 0.5, 0.5);
        float value = 0.75f;
        fprintf(mtl, "\tKa %f %f %f\n", 0.2f, 0.2f, 0.2f);
        fprintf(mtl, "\tKd %f %f %f\n", value, value, value);
        fprintf(mtl, "\tKs %f %f %f\n", 1.0f, 1.0f, 1.0f);
        fprintf(mtl, "\tKe %f %f %f\n", 0.0f, 0.0f, 0.0f);
        fprintf(mtl, "\tillum %d\n", 2);
        fprintf(mtl, "\n");

        fprintf(mtl, "newmtl MAT_BVH_Leaves_%d\n", index);
        fprintf(mtl, "\tNs %f\n", 100.0);
        fprintf(mtl, "\tNi %f\n", 1.5);
        fprintf(mtl, "\td %f\n", 1.0);
        fprintf(mtl, "\tTr %f\n", 0.0);
        fprintf(mtl, "\tTf %f %f %f\n", 0.5, 0.5, 0.5);
        value = 0.1f;
        fprintf(mtl, "\tKa %f %f %f\n", 0.2f, 0.2f, 0.2f);
        fprintf(mtl, "\tKd %f %f %f\n", value, value, value);
        fprintf(mtl, "\tKs %f %f %f\n", 1.0f, 1.0f, 1.0f);
        fprintf(mtl, "\tKe %f %f %f\n", 0.0f, 0.0f, 0.0f);
        fprintf(mtl, "\tillum %d\n", 2);
        fprintf(mtl, "\n");
    }

    fclose(obj);
    fclose(mtl);
    return kr_success;
}
#endif