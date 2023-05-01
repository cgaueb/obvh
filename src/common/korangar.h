#ifndef _KORANGAR_H_
#define _KORANGAR_H_

#ifndef _WIN32
#define KR_PUBLIC_API
#else
#ifdef __cplusplus
#define KR_PUBLIC_API extern "C" __declspec(dllexport)
#else
#define KR_PUBLIC_API __declspec(dllexport)
#endif
#endif

#include "vecmath.h"

//#define KR_DITO_QUANTIZE

typedef struct {
    const char* key;
    const char* value;
} kr_descriptor;

typedef struct {
    kr_descriptor* descriptors;
    kr_size       descriptor_count;
} kr_descriptor_container;

typedef enum {
	KR_TARGET_NONE,
    KR_TARGET_CPU_BUFFER,
    KR_TARGET_CPU_BUFFER_F32,
    KR_TARGET_GL_TEXTURE_2D,
    KR_TARGET_MAX,
} kr_target_type;

typedef struct {
	kr_target_type type;
    union {
        struct {
            kr_u32 width, height;
            kr_u8vec4* data;
        } as_cpu_buffer;
        struct {
            kr_u32 width, height;
            kr_vec4* data;
        } as_cpu_buffer_f32;
        struct {
            kr_u32 width, height;
            kr_u32 handle;
        } as_gl_texture_2d;
    };
} kr_target;

/* Scene management */
typedef enum {
    KR_CAMERA_NONE,
    KR_CAMERA_PINHOLE,
    KR_CAMERA_ORTHO,
    KR_CAMERA_MAX
} kr_camera_type;

typedef enum {
    KR_LIGHT_NONE,
    KR_LIGHT_MESH,
    KR_LIGHT_SPHERE,
    KR_LIGHT_POINT,
    KR_LIGHT_MAX
} kr_light_type;

typedef enum {
    KR_OBJECT_NONE,
    KR_OBJECT_MESH,
    KR_OBJECT_SPHERE,
    KR_OBJECT_AABB,
    KR_OBJECT_MAX
} kr_object_type;

typedef struct {
    kr_transform projection;
    kr_scalar fov;
} kr_camera_pinhole;

typedef struct
{
    kr_camera_type type;
    kr_transform view;
    kr_uvec4 viewport;
    union {
        kr_camera_pinhole as_pinhole;
    };
} kr_camera;

typedef struct {
    kr_vec3 center;
    kr_scalar radius;
} kr_light_sphere;

typedef struct {
    kr_object_type type;
    union {
        kr_light_sphere as_sphere;
    };
} kr_light;

typedef struct {
    kr_vec3* vertices;
    kr_vec3* normals;
    kr_vec2* uvs;
    kr_uvec4* faces;

    kr_size face_count;
    kr_size attr_count;
} kr_mesh;

typedef struct {
    kr_vec3 center;
    kr_scalar radius;
} kr_sphere;

typedef struct {
    kr_aabb3 aabb;
    kr_object_type type;
    union {
        kr_sphere as_sphere;
        kr_mesh   as_mesh;
    };
} kr_object;

typedef struct {
    kr_transform model;
    kr_index object_id;
} kr_object_instance;

typedef enum {
    KR_WRAP_MODE_NONE,
    KR_WRAP_MODE_REPEAT,
    KR_WRAP_MODE_MIRRORED_REPEAT,
    KR_WRAP_MODE_MAX,
} kr_wrap_mode;

typedef struct
{
    kr_wrap_mode wrapS;
    kr_wrap_mode wrapT;
} kr_sampler;

typedef enum {
	KR_TEXTURE_NONE,
	KR_TEXTURE_2D_RGBA8U,
	KR_TEXTURE_2D_RGB8U,
	KR_TEXTURE_2D_RGBA32F,
	KR_TEXTURE_2D_RGB32F,
	KR_TEXTURE_MAX
} kr_texture_type;

typedef struct
{
  kr_u8* data;
  kr_uvec3 dims;
  kr_texture_type type;
} kr_texture;

typedef struct
{
    kr_texture* base_color_texture;
    kr_sampler* base_color_sampler;
    kr_vec3     base_color;
    kr_scalar   roughness;
    kr_scalar   metalicity;
} kr_material;

typedef struct {
    kr_camera camera;
    kr_aabb3 aabb;

    kr_object* objects;
    kr_size object_count;

    kr_object_instance* instances;
    kr_size instance_count;

    kr_camera* cameras;
    kr_size camera_count;

    kr_material* materials;
    kr_size material_count;

    kr_texture* textures;
    kr_size texture_count;

    kr_sampler* samplers;
    kr_size sampler_count;

    kr_uvec2 frame;
    kr_uvec2 tile;

    const char* benchmark_file;
} kr_scene;

/* Next */

#endif /* _KORANGAR_H_ */
