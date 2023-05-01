#ifndef _KORANGAR_GEOMETRY_H_
#define _KORANGAR_GEOMETRY_H_

#include "vecmath.h"
#include "ads.h"

#ifdef __cplusplus
extern "C" {
#endif
	
kr_obb3
kr_points_obb_print(const kr_vec3* points, kr_size count, kr_scalar* elapsed_time);
kr_obb3 
kr_points_obb(const kr_vec3* points, kr_size count, kr_scalar* elapsed_time);
kr_error
kr_bvh_nodes_export(const kr_bvh_node* nodes, const kr_bvh_node* root, const kr_mat4* transforms, const kr_obb3* obbs, i32 level, const char* filename);
kr_error
kr_bvh_node_print(const kr_bvh_node* nodes, const kr_bvh_node* root);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_GEOMETRY_H_ */
