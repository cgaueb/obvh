#ifndef _KORANGAR_SCENE_H_
#define _KORANGAR_SCENE_H_

#include "korangar.h"
#include "vecmath.h"

#ifdef __cplusplus
extern "C" {
#endif

kr_error kr_geometry_cube_create(kr_object* object);
kr_error kr_geometry_triangle_create(kr_object* object);
kr_error kr_geometry_sphere_create(kr_object* object, u32 sector_count, u32 stack_count);
kr_error kr_geometry_gltf_create(kr_object* object, const char* file_path, const char* base_path);
kr_error kr_geometry_plane_create(kr_object* object);
kr_error kr_geometry_move(kr_object* object, vec3 move);
kr_error kr_geometry_transform(kr_object* object, mat4 m);
kr_error kr_geometry_scale(kr_object* object, vec3 scale);
kr_error kr_geometry_scalef(kr_object* object, kr_scalar scale);

kr_error kr_scene_load_settings(kr_scene* scene, const char* filename);
kr_error kr_scene_load(kr_scene* scene, const char* filename);
kr_error kr_scene_export_obj(kr_scene* scene, const char* filename);
kr_error kr_scene_destroy(kr_scene* scene);
kr_error kr_scene_deduplicate(kr_scene* scene);
kr_error kr_scene_transform(kr_scene* scene, mat4 m);

kr_error kr_scene_export_triangles_obj(kr_vec3* vertices, kr_size count, const char* filename);

kr_error
kr_scene_gltf_create(kr_scene* scene, const char* file_path, const char* base_path);
kr_error
kr_scene_gltf_create_flat(kr_scene* scene, const char* file_path, const char* base_path);
kr_error
kr_scene_gltf_create_minecraft(kr_scene* scene);
kr_error
kr_scene_gltf_create_from_bounds(kr_scene* scene, kr_aabb3* bounds, kr_size bounds_count);
kr_error
kr_scene_aabb_calculate(kr_scene* scene);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_SCENE_H_ */
