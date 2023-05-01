#ifdef KR_USE_OPENGL
#define GLEW_NO_GLU
#define GLEW_STATIC
#include <GL/glew.h>
#define WUHOO_OPENGL_ENABLE
#endif
#define WUHOO_IMPLEMENTATION
#include <wuhoo/wuhoo.h>

#define NOMINMAX
#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "common/korangar.h"

#include "common/util.h"
#include "common/logger.h"
#include "common/scene.h"
#include "common/queue.h"
#include "common/integrator.h"
#include "common/ads.h"
#include "common/texture.h"
#include "common/sampling.h"
#include "common/geometry.h"

#include "cmdopt.h"

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define KR_MAX_SCENE_PATH 256
typedef struct {
  char name[KR_MAX_SCENE_PATH];

  kr_integrator  integrators[8];
  kr_integrator* integrator;

  WuhooWindow window;
  WuhooRGBA* pixels;

  const char* benchmark_type;

  kr_target target;
  kr_scene scene;

  b32 mousePressedLastFrame;
  b32 mousePressed;
  vec2 mouse;
  vec2 mouse_last;

  kr_scalar camera_yaw;
  kr_scalar camera_pitch;
  kr_scalar camera_speed;

  u32 width;
  u32 height;

  b32 running;
  b32 save_frame;
  b32 reticle;
} kr_ctx;

#ifdef KR_WIN32
#define KR_MODEL_FOLDER "../../rsrc/gltf/"
#else
#define KR_MODEL_FOLDER "../rsrc/gltf/"
#endif

static
void render_reticle(WuhooRGBA* pixels, i32 width, i32 height, ivec2 at, i32 strech) {
    i32 p = strech;
    for (i32 y = at.y - p; y <= at.y + p; y++) {
        if (y >= height || y < 0) continue;
        u32 index = y * width + at.x;
        pixels[index] = { (u8)(255.0f), (u8)(255.0f), (u8)(255.0f), 0 };
    }
    for (i32 x = at.x - p; x <= at.x + p; x++) {
        if (x >= width || x < 0) continue;
        u32 index = at.y * width + x;
        pixels[index] = { (u8)(255.0f), (u8)(255.0f), (u8)(255.0f), 0 };
    }
}

static
void export_image_f32(u8* pixels, u32 width, u32 height, const char* filename) {
    kr_texture texture = { 0 };
    texture.dims = { width, height, 0 };
    texture.data = pixels;
    kr_texture_save_tiff(&texture, filename, kr_true);
}

static
void export_image(u8* pixels, u32 width, u32 height, const char* filename) {
    kr_texture texture = { 0 };
    texture.dims = { width, height, 0 };
    texture.data = pixels;
    kr_texture_save_png(&texture, filename, kr_true);
}


static
bool render_gl(kr_ctx* ctx) {
#ifdef KR_USE_OPENGL
    ctx->target.type = KR_TARGET_GL_TEXTURE_2D;
    ctx->target.as_gl_texture_2d.width = ctx->width;
    ctx->target.as_gl_texture_2d.height = ctx->height;
    ctx->target.as_gl_texture_2d.handle = 0;

    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    WuhooWindowBlit(&ctx->window, WuhooNull, 0, 0, 0, 0, 0, 0, 0, 0);
#endif

    return true;
}

static
bool render(kr_ctx* ctx) {
    if (KR_EQUALS_LITERAL(ctx->benchmark_type, "diffuse rays sparse")) {
        ctx->target.type = KR_TARGET_CPU_BUFFER_F32;
        ctx->target.as_cpu_buffer_f32.width = ctx->width;
        ctx->target.as_cpu_buffer_f32.height = ctx->height;
        ctx->target.as_cpu_buffer_f32.data = (kr_vec4*)kr_allocate(ctx->width * ctx->height * sizeof(kr_vec4));
    } else {
        ctx->target.type = KR_TARGET_CPU_BUFFER;
        ctx->target.as_cpu_buffer.width = ctx->width;
        ctx->target.as_cpu_buffer.height = ctx->height;
        ctx->target.as_cpu_buffer.data = (kr_u8vec4*)ctx->pixels;
    } 
    render_reticle(ctx->pixels, ctx->width, ctx->height, { (i32)ctx->width / 2, (i32)ctx->height / 2 }, 30);
    WuhooWindowBlit(&ctx->window, ctx->pixels, 0, 0, ctx->width, ctx->height, 0, 0, ctx->width, ctx->height);
    if (ctx->save_frame) {
        export_image((u8*)ctx->pixels, ctx->width, ctx->height, "frame.png");
        ctx->save_frame = kr_false;
    }

    kr_error render_result = kr_integrator_call(ctx->integrator, &ctx->target, KR_ACTION_RENDER);
    if (render_result != kr_success) {
        if (KR_EQUALS_LITERAL(ctx->benchmark_type, "diffuse rays sparse"))
            export_image_f32((u8*)ctx->target.as_cpu_buffer_f32.data, ctx->width, ctx->height, "heatmap.tiff");
        else
            export_image((u8*)ctx->pixels, ctx->width, ctx->height, "frame.png");
        return false;
    }
    return true;
}

int 
main(int argc, char* argv[]) {
  kr_error result = kr_success;
  kr_ctx ctx_placeholder;
  kr_ctx* ctx = &ctx_placeholder;
  kr_zero_memory(ctx, sizeof(*ctx));
  kr_size camera_index = 0;
  b32 fixed_camera = kr_false;
  b32 verbose_output = kr_false;
  b32 flatten_scene = kr_true;
  b32 benchmark_scene = kr_false;
  b32 benchmark_dito = kr_false;
  b32 use_obb = kr_false;
  b32 use_collapse = kr_false;
  b32 diag_rotate = kr_false;
  const char* benchmark_repeat_count = "1";
  const char* benchmark_dito_input = "triangles";
  const char* benchmark_file = "benchmark.csv";
  const char* benhcmark_sample_count = "1000000";
  const char* benhcmark_ray_sample_count = "6";
  const char* benchmark_indirect_bounces = "1";
  const char* benchmark_type = "render";
  const char* integrator_name = "pt_gpu";
  //const char* integrator_name = "bench";
  //const char* intersector_name = "lbvh_gpu";
  const char* intersector_name = "atrbvh_gpu";
  const char* gltf_name = kr_null;

  const char* opt_pattern = 
      "s[1]e[0]f[0]bench[1]v[0]integrator[1]intersector[1]"
      "benchmark_type[1]benchmark_repeat_count[1]"
      "benhcmark_sample_count[1]benhcmark_ray_sample_count[1]"
      "use_obbs[0]dito[0]collapse[0]camera[1]"
      "benchmark_file[1]"
      "benchmark_indirect_bounces[1]"
      "diag_rotate[1]";

    ctx->reticle = kr_true;
    ctx->width = 1024;
    ctx->height = 1024;

  kr_param param;
  kr_opt_get(argc, argv, opt_pattern, "v", &param);
  verbose_output = param.values && param.param_count == 0;
  if (verbose_output) {
      for (int i = 0; i < argc; ++i) {
          //kr_log("Arg[%d] %s\n", i, argv[i]);
      }
  }

    kr_opt_get(argc, argv, opt_pattern, "integrator", &param);
    integrator_name = param.values && param.param_count == 1 ? param.values[1] : integrator_name;

    kr_opt_get(argc, argv, opt_pattern, "intersector", &param);
    intersector_name = param.values && param.param_count == 1 ? param.values[1] : intersector_name;

    kr_opt_get(argc, argv, opt_pattern, "benchmark_type", &param);
    benchmark_type = param.values && param.param_count == 1 ? param.values[1] : benchmark_type;

    kr_opt_get(argc, argv, opt_pattern, "benchmark_file", &param);
    benchmark_file = param.values && param.param_count == 1 ? param.values[1] : benchmark_file;

    kr_opt_get(argc, argv, opt_pattern, "benchmark_repeat_count", &param);
    benchmark_repeat_count = param.values && param.param_count == 1 ? param.values[1] : benchmark_repeat_count;

    kr_opt_get(argc, argv, opt_pattern, "benhcmark_sample_count", &param);
    benhcmark_sample_count = param.values && param.param_count == 1 ? param.values[1] : benhcmark_sample_count;

    kr_opt_get(argc, argv, opt_pattern, "benchmark_indirect_bounces", &param);
    benchmark_indirect_bounces = param.values && param.param_count == 1 ? param.values[1] : benchmark_indirect_bounces;

    kr_opt_get(argc, argv, opt_pattern, "benhcmark_ray_sample_count", &param);
    benhcmark_ray_sample_count = param.values && param.param_count == 1 ? param.values[1] : benhcmark_ray_sample_count;

    kr_opt_get(argc, argv, opt_pattern, "resolution", &param);
    ctx->width = (param.values && param.param_count == 2) ? atoi(param.values[1]) : ctx->width;
    ctx->height = (param.values && param.param_count == 2) ? atoi(param.values[2]) : ctx->height;

    kr_opt_get(argc, argv, opt_pattern, "collapse", &param);
    if (param.values != kr_null) { use_collapse = kr_true; }

    kr_opt_get(argc, argv, opt_pattern, "use_obbs", &param);
    if (param.values != kr_null) { use_obb = kr_true; }

    kr_opt_get(argc, argv, opt_pattern, "dito", &param);
    if (param.values != kr_null) { benchmark_dito = kr_true; }

    kr_opt_get(argc, argv, opt_pattern, "diag_rotate", &param);
    if (param.values != kr_null) { diag_rotate = atoi(param.values[1]); }

    kr_opt_get(argc, argv, opt_pattern, "camera", &param);
    if (param.values != kr_null) { camera_index = atoi(param.values[1]); }

  kr_opt_get(argc, argv, opt_pattern, "s", &param);
  
  gltf_name = param.values[1];

  if (kr_null == gltf_name) return -1;

  kr_opt_get(argc, argv, opt_pattern, "f", &param);

  kr_opt_get(argc, argv, opt_pattern, "bench", &param);
  benchmark_scene = param.values && param.param_count == 1;
  //const char* bench_ray_count_str = param.values[1];
  ctx->benchmark_type = benchmark_type;

  char bench_file_name[128] = { 0 };
  sprintf(bench_file_name, "%s", "bench");
  kr_descriptor options[] = {
      { "benchmark_export_file", bench_file_name },
      { "spp", "1" },
      { "intersector", intersector_name },
      { "reticle", "y" },
      { "save_construction_information", "y" },
      { "ct", "1.0f" },
      { "ci", "1.2f" },
      { "collapse", use_collapse ? "y" : "n" },
     
      { "use_persistent_kernel", "y" },
      { "use_obbs", use_obb ? "y" : "n" },

      // SBVH stuff
      { "sbvh_max_leaf_size", "1" },
  };
  
  kr_descriptor_container settings = { options, sizeof(options) / sizeof(options[0]) };

    const char* title = "Korangar (" WUHOO_PLATFORM_API_STRING ")";

    ctx->mouse = KR_INITIALIZER_CAST(vec2) { (f32)ctx->width / 2.0f, (f32)ctx->height / 2.0f };
    ctx->mouse_last = ctx->mouse;
    ctx->camera_yaw = 0.0f;

    kr_random_engine rng_engine = { (u32)time(kr_null) };
    kr_random_init(&rng_engine);

    kr_size module_index = 0;
    const char* module_names[] = { integrator_name };
    kr_size module_count = (sizeof(module_names)) / (sizeof(module_names[0]));

    for (kr_size module_index = 0; module_index < module_count; module_index++) {
        kr_integrator_load(&ctx->integrators[module_index], module_names[module_index]);
        if (kr_success != result) kr_log("Failed to load integrator '%s'\n", module_names[module_index]);
        else kr_log("Loaded integrator '%s'\n", module_names[module_index]);
    }

    ctx->integrator = &ctx->integrators[module_index];
    if (kr_null == ctx->integrator) return -1;

    kr_log("Using integrator '%s'\n", module_names[module_index]);
    kr_log("Opening scene at %s\n", gltf_name);
    kr_log("Resolution [%d, %d]\n", ctx->width, ctx->height);

    if (!flatten_scene)
        result = kr_scene_gltf_create(&ctx->scene, gltf_name, kr_null);
    else
        result = kr_scene_gltf_create_flat(&ctx->scene, gltf_name, kr_null);

    if (KR_EQUALS_LITERAL(benchmark_type, "ads build") ||
        KR_EQUALS_LITERAL(benchmark_type, "ads sah")) { ctx->scene.benchmark_file = benchmark_file; }

    if (kr_success != result) {
        kr_log("%s returned '%s'\n", gltf_name, result);
        return -1;
    }

    if (kr_success != result) return -1;
    if (kr_null == ctx->scene.objects) return -1;

    kr_scene_aabb_calculate(&ctx->scene);

    vec3 target = kr_aabb_center3(ctx->scene.aabb);
    vec3 origin = kr_vmul31(ctx->scene.aabb.max, 2.2f);
    vec3 front = kr_vnormalize3(kr_vsub3(target, origin));

  vec3 up = (kr_vdot3(front, { 0, 1, 0 }) < -0.9999f) ? vec3{ 0, 0, -1 } : vec3{ 0, 1, 0 };
  vec3 right  = kr_vnormalize3(kr_vcross3(front, up));
  up = kr_vnormalize3(kr_vcross3(right, front));

  fixed_camera = (ctx->scene.camera.type == KR_CAMERA_PINHOLE);

  ctx->scene.camera.type = KR_CAMERA_PINHOLE;
  ctx->scene.camera.viewport = uvec4{ 0, 0, 1000, 1000 };
  ctx->scene.camera.view = kr_mtransform4(kr_mlookat4(origin, kr_vadd3(origin, front), up));
  ctx->scene.camera.as_pinhole.fov = kr_radians(60.0f);
  ctx->scene.camera.as_pinhole.projection = kr_mtransform4(kr_mperspective4(ctx->scene.camera.as_pinhole.fov, (kr_scalar)ctx->scene.camera.viewport.z / (kr_scalar)ctx->scene.camera.viewport.w, 1.0f, 1000.0f));

  ctx->width = ctx->scene.camera.viewport.z;
  ctx->height = ctx->scene.camera.viewport.w;
 
  up = kr_vto43(ctx->scene.camera.view.from.cols[1]);
  front = kr_vto43(ctx->scene.camera.view.from.cols[2]);
  origin = kr_vto43(ctx->scene.camera.view.from.cols[3]);
  target = kr_vadd3(origin, front);

  ctx->scene.camera.view = kr_mtransform4(kr_mlookat4(origin, front, up));

  front = kr_vnormalize3(kr_vsub3(target, origin));
  right = kr_vnormalize3(kr_vcross3(front, up));

  ctx->camera_pitch = (kr_scalar)kr_degrees((kr_scalar)asin(front.y));
  ctx->camera_yaw = (kr_scalar)kr_degrees((kr_scalar)acos(front.x / cos(kr_radians(ctx->camera_pitch))));
  ctx->camera_speed = kr_vlength3(kr_aabb_extents3(ctx->scene.aabb)) * 0.01f;
  ctx->pixels = (WuhooRGBA*)kr_allocate(ctx->width * ctx->height * sizeof(*ctx->pixels));
  ctx->mouse_last = { -1.0f, -1.0f };
  ctx->mouse = { -1.0f, -1.0f };
  ctx->mousePressed = 0;
  ctx->mousePressedLastFrame = 0;

  kr_zero_memory(ctx->pixels, ctx->width* ctx->height * sizeof(*ctx->pixels));

    WuhooWindowInit(&ctx->window);
    WuhooWindowCreate(&ctx->window, WuhooDefaultPosition, WuhooDefaultPosition, ctx->width, ctx->height, title,
        WUHOO_FLAG_CANVAS |
        WUHOO_FLAG_TITLED |
        WUHOO_FLAG_RESIZEABLE |
        WUHOO_FLAG_CLIENT_REGION |
#ifdef KR_USE_OPENGL
        WUHOO_FLAG_OPENGL |
#endif
        WUHOO_FLAG_CLOSEABLE, WuhooNull);
    WuhooWindowShow(&ctx->window);

#ifdef KR_USE_OPENGL
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        printf("|Mandelbrot GL| glewInit failed with %s\n", glewGetErrorString(err));
        return 1;
    }
#endif

  result = kr_integrator_call(ctx->integrator, &settings, KR_ACTION_INIT);
  result = kr_integrator_call(ctx->integrator, &ctx->scene, KR_ACTION_COMMIT);

  WuhooEvent event = { };
  ctx->running = kr_true;
  kr_scalar angle = 0.0f;
  while (kr_true == ctx->running) {
    WuhooWindowEventNext(&ctx->window, &event);

    switch (event.type) {
    case WUHOO_EVT_WINDOW:
        switch (event.data.window.state) {
        case WUHOO_WSTATE_INVALIDATED:
            WuhooWindowBlit(&ctx->window, ctx->pixels, 0, 0, ctx->width, ctx->height, 0, 0, ctx->width, ctx->height);

            break;
        case WUHOO_WSTATE_CLOSED:
            ctx->running = kr_false;
            break;
        default:
            break;
        }
        break;
    case WUHOO_EVT_MOUSE_PRESS:
    {
        switch (event.data.mouse_press.state)
        {
        case WUHOO_MSTATE_LPRESSED:
        {
            ctx->mousePressed = 1;
            ctx->mousePressedLastFrame = 1;
        }break;
        case WUHOO_MSTATE_LRELEASED:
        {
            ctx->mousePressed = 0;
            ctx->mousePressedLastFrame = 0;
        }break;
        }
    }break;
    case WUHOO_EVT_MOUSE_MOVE:
    {
        if (ctx->mousePressedLastFrame)
        {
            ctx->mouse_last = { (kr_scalar)event.data.mouse_move.x, (kr_scalar)event.data.mouse_move.y };
            ctx->mousePressedLastFrame = 0;
        }

        if (ctx->mousePressed)
        {
            ctx->mouse = { (kr_scalar)event.data.mouse_move.x, (kr_scalar)event.data.mouse_move.y };

            const kr_scalar dx = ctx->mouse_last.x - ctx->mouse.x;
            const kr_scalar dy = ctx->mouse_last.y - ctx->mouse.y;
            const kr_scalar radX = 0.001 * dx * KR_PI;
            const kr_scalar radY = 0.001 * dy * KR_PI;

            const mat4 rot1 = kr_mrotate4(up, radX);
            const mat4 rot2 = kr_mrotate4(right, radY);
            const vec3 rotated_front = kr_mvmul43(rot2, kr_mvmul43(rot1, front));
            const kr_scalar distToTar = kr_vdistance3(origin, target);
            target = kr_vadd3(origin, kr_vmul31(rotated_front, distToTar));
            ctx->mouse_last = ctx->mouse;
        }
    }break;
    case WUHOO_EVT_MOUSE_WHEEL:
        origin = kr_vadd3(origin, kr_vmul31(front, event.data.mouse_wheel.delta_y * ctx->camera_speed));
        target = kr_vadd3(target, kr_vmul31(front, event.data.mouse_wheel.delta_y * ctx->camera_speed));
        break;
    case WUHOO_EVT_KEY:
        switch (event.data.key.code) {
        case WUHOO_VKEY_ESCAPE:
            ctx->running = (event.data.key.state == WUHOO_KSTATE_UP);
            break;
        case WUHOO_VKEY_LEFT:
        case WUHOO_VKEY_A:
            origin = kr_vsub3(origin, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(right, ctx->camera_speed) : vec3{ 0, 0, 0 });
            target = kr_vsub3(target, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(right, ctx->camera_speed) : vec3{ 0, 0, 0 });
            break;
        case WUHOO_VKEY_RIGHT:
        case WUHOO_VKEY_D:
            origin = kr_vadd3(origin, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(right, ctx->camera_speed) : vec3{ 0, 0, 0 });
            target = kr_vadd3(target, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(right, ctx->camera_speed) : vec3{ 0, 0, 0 });
            break;
        case WUHOO_VKEY_UP:
        case WUHOO_VKEY_W:
            //origin = kr_vadd3(origin, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(up, ctx->camera_speed) : vec3{ 0, 0, 0 });
            origin = kr_vadd3(origin, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(up, ctx->camera_speed) : vec3{ 0, 0, 0 });
            target = kr_vadd3(target, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(up, ctx->camera_speed) : vec3{ 0, 0, 0 });
            break;
        case WUHOO_VKEY_DOWN:
        case WUHOO_VKEY_S:
            //origin = kr_vsub3(origin, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(up, ctx->camera_speed) : vec3{ 0, 0, 0 });
            origin = kr_vsub3(origin, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(up, ctx->camera_speed) : vec3{ 0, 0, 0 });
            target = kr_vsub3(target, (event.data.key.state == WUHOO_KSTATE_DOWN) ? kr_vmul31(up, ctx->camera_speed) : vec3{ 0, 0, 0 });
            break;
        case WUHOO_VKEY_E:
            if (WUHOO_KSTATE_UP == event.data.key.state) {
                kr_ads_csv_export export_csv = { &ctx->name[0], "." };
                kr_ads_action_payload payload = {KR_ADS_ACTION_CSV_EXPORT, &export_csv };
                kr_integrator_call(ctx->integrator, &payload, KR_ACTION_ADS_ACTION);
            }
            break;
        case WUHOO_VKEY_P:
            ctx->save_frame = kr_true;
        case WUHOO_VKEY_T:
            if (WUHOO_KSTATE_UP == event.data.key.state) {
                kr_ads_bvh_model_export export_bvh = { "bvh_bounds", "."};
                kr_ads_action_payload payload = { KR_ADS_ACTION_BVH_MODEL_EXPORT, &export_bvh };
                kr_integrator_call(ctx->integrator, &payload, KR_ACTION_ADS_ACTION);
            }
            break;
        default:
            break;
        }
    }

    ctx->scene.camera.view = kr_mtransform4(kr_mlookat4(origin, target, { 0, 1, 0 }));
    front = kr_vnormalize3(kr_vsub3(target, origin));
    right = kr_vnormalize3(kr_vcross3(front, up));
    ctx->scene.camera.view = kr_mtransform4(kr_mlookat4(origin, target, { 0, 1, 0 }));

    kr_error update_result = kr_integrator_call(ctx->integrator, &ctx->scene.camera, KR_ACTION_CAMERA_UPDATE);

#ifdef KR_USE_OPENGL
    render_gl(ctx);
#else
    if (!render(ctx))
        ctx->running = kr_false;
#endif
  }

  kr_integrator_call(ctx->integrator, &ctx->scene, KR_ACTION_DESTROY);

  kr_scene_destroy(&ctx->scene);
  WuhooWindowDestroy(&ctx->window);

  kr_free((void**)&ctx->pixels);

#if defined(_DEBUG) && defined(KR_WIN32)
  //_CrtDumpMemoryLeaks();
#endif
  
  return 0;
}
