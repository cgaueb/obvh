#define NOMINMAX
#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "bench.h"

#include "common/util.h"
#include "common/cuda/util.h"
#include "common/integrator.h"
#include "common/ads.h"
#include "common/scene.h"
#include "common/geometry.h"
#include "common/logger.h"
#include "common/algorithm/dito/dito.cuh"

#include <array>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

kr_integrator_bench*
integrator_bench_create() {
    return (kr_integrator_bench*) kr_allocate(sizeof(kr_integrator_bench));
}

kr_error
integrator_bench_update(kr_integrator_bench* ao, kr_scene* scene) {
   
    return kr_success;
}

kr_error
integrator_bench_commit(kr_integrator_bench* bench, kr_scene* scene) {
    bench->scene = scene;

    return kr_success;
}

kr_error
integrator_bench_init(kr_integrator_bench* bench, kr_descriptor_container* settings) {
    const char* intersector = "lbvh";
    bench->benchmark_repeat_count = 1;
    bench->sample_count = 1000000;
    bench->ray_sample_count = 1;

    for (kr_size i = 0; i < settings->descriptor_count; i++) {
        if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benhcmark_sample_count")) {
            bench->sample_count = (u32)atoi(settings->descriptors[i].value);
        } else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benhcmark_ray_sample_count")) {
            bench->ray_sample_count = (u32)atoi(settings->descriptors[i].value);
        } else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_dito")) {
            bench->benchmark_dito = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_dito")) {
            bench->benchmark_dito = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_dito_input")) {
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_file")) {
            bench->benchmark_file = settings->descriptors[i].value;
        } else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_type")) {
            bench->benchmark_type = settings->descriptors[i].value;
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "intersector")) {
            intersector = settings->descriptors[i].value;
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_repeat_count")) {
            bench->benchmark_repeat_count = (u32)atoi(settings->descriptors[i].value);
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_export_ray_origins")) {
            bench->export_sampled_origins = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_export_ray_directions")) {
            bench->export_sampled_directions = KR_EQUALS_LITERAL(settings->descriptors[i].value, "y");
        }
        else if (KR_EQUALS_LITERAL(settings->descriptors[i].key, "benchmark_indirect_bounces")) {
            bench->indirect_bounces = (u32)atoi(settings->descriptors[i].value);
        }
    }

    kr_random_engine rng_engine = { (u32)time(kr_null) };
    kr_random_init(&rng_engine);
    bench->rng = rng_engine;

    bench->frame = 0;

    return kr_success;
}

#define KR_RENDER_DONE ((kr_error)"Simulation is done")
kr_internal kr_error
integrator_bench_dito(kr_integrator_bench* bench) {
    kr_scene* scene = bench->scene;
    kr_u32 instance_index = 0;
    kr_object_instance* instance = &scene->instances[instance_index];
    kr_object* object = &scene->objects[instance->object_id];

    kr_size primitive_count = object->as_mesh.face_count;

    kr_size sample_count = object->as_mesh.attr_count;
    vec3* points = (vec3*)kr_allocate(sizeof(*points) * sample_count);
    memcpy(points, object->as_mesh.vertices, sizeof(vec3) * sample_count);

    kr_scalar cpu_timer = 0.0f;
    kr_obb3 obb_cpu = kr_points_obb(points, sample_count, &cpu_timer);

    kr_obb3 obb = obb_cpu;
    printf("Computed OBB CPU:\n");
    printf("Midpoint: %f %f %f\n", obb.mid.x, obb.mid.y, obb.mid.z);
    printf("v0: %f %f %f\n", obb.v0.x, obb.v0.y, obb.v0.z);
    printf("v1: %f %f %f\n", obb.v1.x, obb.v1.y, obb.v1.z);
    printf("v2: %f %f %f\n", obb.v2.x, obb.v2.y, obb.v2.z);
    printf("ext: %f %f %f\n", obb.ext.x, obb.ext.y, obb.ext.z);
    printf("Area: %f\n", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));

    kr_scalar gpu_timer = 0.0f;
    kr_obb3 obb_gpu = kr_cuda_points_obb(points, sample_count, &gpu_timer);

    obb = obb_gpu;
    printf("Computed OBB GPU:\n");
    printf("Midpoint: %f %f %f\n", obb.mid.x, obb.mid.y, obb.mid.z);
    printf("v0: %f %f %f\n", obb.v0.x, obb.v0.y, obb.v0.z);
    printf("v1: %f %f %f\n", obb.v1.x, obb.v1.y, obb.v1.z);
    printf("v2: %f %f %f\n", obb.v2.x, obb.v2.y, obb.v2.z);
    printf("ext: %f %f %f\n", obb.ext.x, obb.ext.y, obb.ext.z);
    printf("Area: %f\n", 2 * (obb.ext.x * obb.ext.y + obb.ext.x * obb.ext.z + obb.ext.y * obb.ext.z));

    kr_free((void**)&points);

    return KR_RENDER_DONE;
}

kr_error
integrator_bench_render(kr_integrator_bench* bench, kr_target* target) {
    return integrator_bench_dito(bench);
}