#ifndef _KORANGAR_SCENE_CUDA_H_
#define _KORANGAR_SCENE_CUDA_H_

#include "common/korangar.h"

typedef struct {
    kr_vec3*  vertices;
    kr_size attr_count;
    kr_uvec4* faces;
    kr_size face_count;
} kr_mesh_cu;

typedef struct {
  kr_aabb3 aabb;
  kr_object_type type;
  union {
      kr_sphere   as_sphere;
      kr_mesh_cu  as_mesh;
  };
} kr_object_cu;

typedef struct {
    aabb3 aabb;
    aabb3 centroid_aabb;

    kr_camera* camera;

    kr_object_cu* objects;
    kr_size object_count;

    kr_object_instance* instances;
    kr_size instance_count;
} kr_scene_cu;

#endif /* _KORANGAR_SCENE_CUDA_H_ */