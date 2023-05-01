#ifndef _KORANGAR_BENCH_H_
#define _KORANGAR_BENCH_H_

#include "common/korangar.h"
#include "common/ads.h"
#include "common/util.h"
#include "common/sampling.h"

typedef struct {
    kr_ads adses[8];
    kr_random_engine rng;
    kr_piecewise_constant_1D triangle_dist;
    const char* benchmark_file;
    const char* benchmark_type;
    kr_ads* ads;

    kr_scene* scene;

    u32 sample_count;
    u32 ray_sample_count;
    u32 frame;
    u32 version;
    u32 benchmark_repeat_count;
    u32 indirect_bounces;

    b32 benchmark_dito;
    b32 export_sampled_origins;
    b32 export_sampled_directions;
    b32 export_mismatches;
    b32 count_mismatches;
} kr_integrator_bench;

#ifdef __cplusplus
extern "C" {
#endif

kr_error
integrator_bench_render(kr_integrator_bench* ao, kr_target* target);
kr_error
integrator_bench_init(kr_integrator_bench* ao, kr_descriptor_container* settings);
kr_error
integrator_bench_commit(kr_integrator_bench* ao, kr_scene* scene);
kr_integrator_bench*
integrator_bench_create();
kr_error
integrator_bench_update(kr_integrator_bench* ao, kr_scene* scene);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_BENCH_H_ */