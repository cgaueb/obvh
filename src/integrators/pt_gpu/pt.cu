#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "common/integrator.h"
#include "common/ads.h"
#include "common/util.h"
#include "common/cuda/util.h"
#include "common/cuda/scene.h"

#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

typedef enum {
    KR_UPDATE_NONE = 0,
    KR_UPDATE_SCENE_GEOMETRY = kr_flag(0),
    KR_UPDATE_MAX
} kr_update_flags;

typedef struct {
    kr_ads adses[8];
    kr_ads* ads;
    
    kr_vec4* h_framebuffer;
    kr_vec4* d_framebuffer;
    kr_scene_cu* h_scene;
    kr_scene_cu* d_scene;
    const kr_scene* scene;

    kr_camera* camera;
    kr_ray* rays;
    kr_intersection* d_isects;
    kr_intersection* h_isects;

    kr_update_flags flags;
    u32 version;
    u32 ray_count;
    u32 sample_count;
    u32 frame_counter;
} kr_integrator_pt;

kr_error
integrator_pt_cuda_render(kr_integrator_pt* pt, kr_target* target);
kr_error
integrator_pt_cuda_update(kr_integrator_pt* pt, kr_scene* scene);
kr_error
integrator_pt_cuda_init(kr_integrator_pt* pt, kr_descriptor_container* settings);
kr_error
integrator_pt_cuda_commit(kr_integrator_pt* pt, kr_scene* scene);
kr_error
integrator_pt_cuda_camera_update(kr_integrator_pt* pt, const kr_camera* camera);
kr_integrator_pt*
integrator_pt_cuda_create();

kr_integrator_pt*
integrator_pt_cuda_create() {
  return (kr_integrator_pt*) kr_allocate(sizeof(kr_integrator_pt));
}

kr_error
integrator_pt_cuda_camera_update(kr_integrator_pt* pt, const kr_camera* camera) {
    cudaMemcpy(pt->h_scene->camera, camera, sizeof(*camera), cudaMemcpyHostToDevice);

    return kr_success;
}

kr_error
integrator_pt_cuda_init(kr_integrator_pt* pt, kr_descriptor_container* settings) {
    const char* intersector = "lbvh";
    for (kr_size i = 0; i < settings->descriptor_count; i++) {
        if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "intersector")) {
            intersector = settings->descriptors[i].value;
        }
    }
   
    kr_ads_load(&pt->adses[0], intersector);
    pt->ads = &pt->adses[0];

    kr_ads_call(pt->ads, settings, KR_ADS_ACTION_INIT);

    return kr_success;
}

kr_error
integrator_pt_cuda_commit(kr_integrator_pt* pt, kr_scene* scene) {
  const kr_camera* camera = &scene->camera;

  kr_size object_count = scene->object_count;
  kr_size instance_count = scene->instance_count;

  pt->scene = scene;

  kr_scene_cu* h_scene_cu = (kr_scene_cu*)kr_allocate(sizeof(*h_scene_cu));

  h_scene_cu->objects = (kr_object_cu*)kr_cuda_allocate(scene->object_count * sizeof(*h_scene_cu->objects));
  h_scene_cu->instances = (kr_object_instance*)kr_cuda_allocate(scene->instance_count * sizeof(*h_scene_cu->instances));
  
  h_scene_cu->camera = (kr_camera*)kr_cuda_allocate(sizeof(*camera));
  cudaMemcpy(h_scene_cu->camera, camera, sizeof(*camera), cudaMemcpyHostToDevice);

  pt->d_framebuffer = (kr_vec4*)kr_cuda_allocate(camera->viewport.z * camera->viewport.w * sizeof(*pt->d_framebuffer));
  pt->h_framebuffer = (kr_vec4*)kr_allocate(camera->viewport.z * camera->viewport.w * sizeof(*pt->h_framebuffer));

  kr_size primitive_count = 0;

  for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
      kr_object_instance* instance = &scene->instances[instance_index];
      kr_object* object = &scene->objects[instance->object_id];
      if (KR_OBJECT_NONE != object->type) {
          continue;
      }

      kr_object_cu  h_object_cu = { 0 };
      kr_object_cu* d_object_cu = (kr_object_cu*)kr_cuda_allocate(sizeof(*d_object_cu));
      h_object_cu.type = object->type;

      switch (object->type) {
      case KR_OBJECT_AABB:
          primitive_count = 1;
          break;
      case KR_OBJECT_MESH: {
          kr_u32 face_count = (kr_u32)object->as_mesh.face_count;
          kr_u32 attr_count = (kr_u32)object->as_mesh.attr_count;
          primitive_count = (kr_u32)object->as_mesh.face_count;

          h_object_cu.as_mesh.vertices = (vec3*)kr_cuda_allocate(object->as_mesh.attr_count * sizeof(*h_object_cu.as_mesh.vertices));
          h_object_cu.as_mesh.attr_count = object->as_mesh.attr_count;
          h_object_cu.as_mesh.faces = (uvec4*)kr_cuda_allocate(object->as_mesh.face_count * sizeof(*h_object_cu.as_mesh.faces));
          h_object_cu.as_mesh.face_count = object->as_mesh.face_count;

          cudaMemcpy(h_object_cu.as_mesh.vertices, object->as_mesh.vertices, attr_count * sizeof(*object->as_mesh.vertices), cudaMemcpyHostToDevice);
          cudaMemcpy(h_object_cu.as_mesh.faces, object->as_mesh.faces, face_count * sizeof(*object->as_mesh.faces), cudaMemcpyHostToDevice);

          break;
      }
      default:
          break;
      }

      cudaMemcpy(d_object_cu, &h_object_cu, sizeof(h_object_cu), cudaMemcpyHostToDevice);
  }

  h_scene_cu->object_count   = object_count;
  h_scene_cu->instance_count = instance_count;

  kr_scene_cu* d_scene_cu = (kr_scene_cu*)kr_cuda_allocate(sizeof(*d_scene_cu));
  cudaMemcpy(d_scene_cu, h_scene_cu, sizeof(*h_scene_cu), cudaMemcpyHostToDevice);

  pt->h_scene = h_scene_cu;
  pt->d_scene = d_scene_cu;

    kr_ads_call(pt->ads, scene, KR_ADS_ACTION_COMMIT);

    return kr_success;
}


__global__
void generate(kr_ray* rays, const kr_camera* camera)
{
    const i32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= camera->viewport.z || y >= camera->viewport.w)
        return;
    const i32 ray_index = y * camera->viewport.z + x;
    kr_ray* ray = &rays[ray_index];
   
    vec2 screen = KR_INITIALIZER_CAST(vec2){1.0 - (float)x / (float)camera->viewport.z, 1.0f - (float)y / (float)camera->viewport.w};
    vec3 ndc = KR_INITIALIZER_CAST(vec3){ screen.x * 2.0f - 1.0f, screen.y * 2.0f - 1.0f, 0.0f };
    vec3 view = kr_vproject3(camera->as_pinhole.projection.from, ndc);

    vec3 direction = kr_vto43(camera->view.from.cols[2]);
    vec3 origin = kr_vto43(camera->view.from.cols[3]);
    switch (camera->type) {
    case KR_CAMERA_PINHOLE:
        direction = kr_vnormalize3(view);
        origin = kr_vtransform3(camera->view.from, KR_INITIALIZER_CAST(vec3) { 0, 0, 0 });
        direction = kr_ntransform3(camera->view.from, direction);
        break;
    case KR_CAMERA_ORTHO:
        origin = kr_vtransform3(camera->view.from, KR_INITIALIZER_CAST(vec3) { ndc.x, ndc.y, 0 });
        break;
    }
    ray->origin = origin;
    ray->tmin = 0.01;
    ray->direction = kr_vnormalize3(direction);
    ray->tmax = FLT_MAX;
}

typedef union {
    u32 s[4];
} kr_cuda_random_engine;

kr_inline_device static inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

kr_inline_device uint32_t next(kr_cuda_random_engine* rng) {
    const uint32_t result = rng->s[0] + rng->s[3];

    const uint32_t t = rng->s[1] << 9;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;

    rng->s[3] = rotl(rng->s[3], 11);

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations. */

kr_inline_device void jump(kr_cuda_random_engine* rng) {
    static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for (int i = 0; i < sizeof JUMP / sizeof * JUMP; i++)
        for (int b = 0; b < 32; b++) {
            if (JUMP[i] & UINT32_C(1) << b) {
                s0 ^= rng->s[0];
                s1 ^= rng->s[1];
                s2 ^= rng->s[2];
                s3 ^= rng->s[3];
            }
            next(rng);
        }

    rng->s[0] = s0;
    rng->s[1] = s1;
    rng->s[2] = s2;
    rng->s[3] = s3;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^96 calls to next(); it can be used to generate 2^32 starting points,
   from each of which jump() will generate 2^32 non-overlapping
   subsequences for parallel distributed computations. */

kr_inline_device void long_jump(kr_cuda_random_engine* rng) {
    static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for (int i = 0; i < sizeof LONG_JUMP / sizeof * LONG_JUMP; i++)
        for (int b = 0; b < 32; b++) {
            if (LONG_JUMP[i] & UINT32_C(1) << b) {
                s0 ^= rng->s[0];
                s1 ^= rng->s[1];
                s2 ^= rng->s[2];
                s3 ^= rng->s[3];
            }
            next(rng);
        }

    rng->s[0] = s0;
    rng->s[1] = s1;
    rng->s[2] = s2;
    rng->s[3] = s3;
}

kr_inline_device f32 FloatFromBits(const u32 i) {
    return (i >> 8) * 0x1.0p-24f;
}

kr_inline_device void kr_cuda_random_init(kr_cuda_random_engine* rng, u32 seed) {
    rng->s[0] = seed;
    rng->s[1] = seed * seed;
    rng->s[2] = seed * seed * seed * seed;
    rng->s[3] = seed * seed * seed * seed * seed * seed * seed * seed;
}

kr_inline_device void
kr_randomf(kr_cuda_random_engine* rng, kr_scalar* rands, u32 count) {
    for (u32 i = 0; i < count; i++) {
        rands[i] = FloatFromBits(next(rng));
    }
}

kr_inline_device kr_vec4
kr_vrandom4(kr_cuda_random_engine* rng) {
    kr_vec4 v;
    kr_randomf(rng, v.v, 4);
    return v;
}

__global__
void render(kr_vec4* frame, uvec4 viewport, u32 frame_counter, u32 seed)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = r * viewport.z + c; // 1D flat index
    const float value = 1.0f - (float)c / (float)viewport.z;

    kr_cuda_random_engine rng;
    kr_cuda_random_init(&rng, i ^ seed);

    vec4 new_value = kr_vrandom4(&rng);
    vec4 old_value = frame[i];

    frame[i] = kr_vlerp4(old_value, new_value, (kr_scalar)1 / (kr_scalar)frame_counter);
    //frame[i] = {1,1,1,1};
}

kr_error
integrator_pt_cuda_render(kr_integrator_pt* pt, kr_target* target) {
    const kr_camera* camera = &pt->scene->camera;
    u32 height = target->as_cpu_buffer.height;
    u32 width = target->as_cpu_buffer.width;
    u32 samples = 1;
    u32 ray_count = width * height * samples;
    u8vec4* pixels = target->as_cpu_buffer.data;

    int device = -1;
    cudaGetDevice(&device);
    if (pt->ray_count != ray_count) {
        pt->rays = (kr_ray*)kr_cuda_allocate(ray_count * sizeof(*pt->rays));
        pt->d_isects = (kr_intersection*)kr_cuda_allocate(ray_count * sizeof(*pt->d_isects));
        pt->h_isects = (kr_intersection*)kr_allocate(ray_count * sizeof(*pt->h_isects));
        pt->ray_count = ray_count;
    }

    {
        dim3 blockSize = dim3(32, 32);
        int bx = (width + blockSize.x - 1) / blockSize.x;
        int by = (height + blockSize.y - 1) / blockSize.y;
        dim3 gridSize = dim3(bx, by);
        generate <<< gridSize, blockSize >>> (pt->rays, pt->h_scene->camera);
    }

    kr_intersection_query query = { KR_QUERY_TYPE_CUDA, {pt->rays, pt->d_isects, pt->ray_count} };


    kr_scalar elapsed_ms = KernelLaunch().execute([&]() {
        kr_ads_call(pt->ads, &query, KR_ADS_ACTION_INTERSECTION_QUERY);
    });
    //printf("Intersection calculation took %fms\n", elapsed_ms);

    u32 tom = (u32)rand();
    {
        pt->frame_counter++;
        dim3 blockSize = dim3(32, 32);
        int bx = (width + blockSize.x - 1) / blockSize.x;
        int by = (height + blockSize.y - 1) / blockSize.y;
        dim3 gridSize = dim3(bx, by);
        //render <<< gridSize, blockSize >>> (pt->d_framebuffer, camera->viewport, pt->frame_counter, tom);
    }

    cudaMemcpy(pt->h_isects, pt->d_isects, ray_count * sizeof(*pt->d_isects), cudaMemcpyDeviceToHost);
    //cudaMemcpy(pt->h_framebuffer, pt->d_framebuffer, width * height * sizeof(*pt->h_framebuffer), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++) {
        //pixels[i] = kr_vrgba4(pt->h_framebuffer[i]);
        pixels[i] = { 0, 0, 0, 0 };
        if (pt->h_isects[i].primitive != kr_invalid_index) {
            pixels[i] = kr_vrgba3(kr_vabs3(pt->h_isects[i].geom_normal));
            //pixels[i] = kr_vrgba2(pt->isects[i].barys);
        }
        pixels[i].a = 255;
    }

    return kr_success;
}

extern "C" KR_PUBLIC_API kr_error 
korangar_action(kr_integrator* integrator, kr_handle descriptor, kr_action action) {
    
	switch (action) {
    case KR_ACTION_CREATE:
        return (kr_error) integrator_pt_cuda_create();
        break;
    case KR_ACTION_RENDER:
        return integrator_pt_cuda_render((kr_integrator_pt*)integrator->context, (kr_target*)descriptor);
        break;
    case KR_ACTION_INIT:
        return integrator_pt_cuda_init((kr_integrator_pt*)integrator->context, (kr_descriptor_container*)descriptor);
        break;
    case KR_ACTION_COMMIT:
        return integrator_pt_cuda_commit((kr_integrator_pt*)integrator->context, (kr_scene*)descriptor);
        break;
    case KR_ACTION_UPDATE:
        //return integrator_pt_cuda_update((kr_integrator_pt*)integrator->context, (kr_scene*)descriptor);
        break;
    case KR_ACTION_CAMERA_UPDATE:
        return integrator_pt_cuda_camera_update((kr_integrator_pt*)integrator->context, (kr_camera*)descriptor);
        break;
    default:
        return kr_success;
        break;
	}
    return kr_success;
}