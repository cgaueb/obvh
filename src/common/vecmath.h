#ifndef KR_VECMATH_H
#define KR_VECMATH_H

#include <math.h>
#include <float.h>

#if defined(__CUDACC__)
//#include <vector_types.h>
#endif
//#define KR_DITO_QUANTIZE
#include <intrin.h>

#ifdef _DEBUG
#ifndef KR_DEBUG
#define KR_DEBUG
#endif
#endif

#define KR_LEAN_TYPES
#define KR_DEBUG_FACILITIES
#define KR_ERROR(err) ((kr_error)(err))
#define KR_BOOLSTR(b) ((b)?"true":"false")

#define kr_success kr_null
#define kr_invalid_index ((kr_u32)-1)
#define kr_internal static
#define kr_true (1)
#define kr_false (0)
#define kr_offsetof(st, m) \
    ((kr_size)&(((st *)0)->m))
#define kr_flag(x) (1 << (x))

#ifdef _WIN32 
#define KR_WIN32 1
#endif

#ifdef __linux__ 
#define KR_LINUX 1
#endif

typedef unsigned long long kr_u64;
typedef long long          kr_i64;
typedef unsigned int       kr_u32;
typedef int                kr_i32;
typedef int                kr_b32;
typedef unsigned char      kr_u8;
typedef char               kr_i8;
typedef unsigned short     kr_u16;
typedef short              kr_i16;
typedef int                kr_b32;
typedef float              kr_f32;
typedef double             kr_f64;

#ifdef KR_LEAN_TYPES
typedef unsigned long long u64;
typedef long long          i64;
typedef unsigned int       u32;
typedef int                i32;
typedef unsigned char      u8;
typedef char               i8;
typedef unsigned short     u16;
typedef short              i16;
typedef int                b32;
typedef float              f32;
typedef double             f64;
#endif

typedef u64                kr_size;
typedef i64                kr_isize;
typedef kr_size            kr_index;
typedef kr_size            kr_offset;
typedef const void* kr_handle;
typedef const void* kr_error;

//#define KR_DOUBLE_PRECISION
#ifndef KR_DOUBLE_PRECISION
typedef f32                kr_scalar;
#define KR_RAY_TRAVEL_ZERO  -FLT_MAX
#define KR_RAY_TRAVEL_INF   FLT_MAX
#else
typedef f64                kr_scalar;
#define KR_RAY_TRAVEL_ZERO  -DBL_MAX
#define KR_RAY_TRAVEL_INF    DBL_MAX
#endif

#ifdef __cplusplus
#define KR_INITIALIZER_CAST(type)
#define kr_null nullptr
#define kr_constexpr constexpr
#else
#define KR_INITIALIZER_CAST(type) (type)
#define kr_null ((void*)0)
#define kr_constexpr
#endif

#if defined(__CUDACC__)
#define kr_device __device__
#define kr_host __host__
#define kr_inline inline
#define kr_inline_device inline __device__
#define kr_inline_host_device inline __host__ __device__
#define kr_global __global__
#define kr_align(x)
#define KR_CUDA 1
#else
#define KR_CUDA 0
#define kr_device
#ifdef __cplusplus
#define kr_inline_device __forceinline
#define kr_inline_host_device kr_inline_device
#else
#define kr_inline_device extern inline
#define kr_inline_host_device kr_inline_device
#endif
#if defined(__GNUC__) || defined(__clang__)
#  define kr_align(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#  define kr_align(x) __declspec(align(x))
#else
#  error "Unknown compiler; can't define ALIGN"
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define kr_align_of(X) __alignof__(X)
#elif defined(_MSC_VER)
#    define kr_align_of(X) __alignof(X)
#else
#  error "Unknown compiler; can't define ALIGNOF"
#endif

#endif

#if 1//!defined(__CUDACC__)
/* Vector Math Types */
typedef union { kr_scalar v[2]; struct { kr_scalar x, y; }; struct { kr_scalar r, g; }; } vec2, kr_vec2;
typedef union { kr_scalar v[3]; struct { kr_scalar x, y, z; }; struct { kr_scalar r, g, b; }; struct { kr_vec2 xy; }; } vec3, kr_vec3;
typedef union { kr_scalar v[4]; struct { kr_scalar x, y, z, w; }; struct { kr_scalar r, g, b, a; }; } vec4, kr_vec4;
typedef vec4 quat;
#else
typedef float2 kr_vec2;
typedef float3 kr_vec3;
typedef float4 kr_vec4;
typedef kr_vec2 vec2;
typedef kr_vec3 vec3;
typedef kr_vec4 vec4;
#endif
typedef const kr_vec2 cvec2;
typedef const kr_vec3 cvec3;
typedef const kr_vec4 cvec4;

typedef const kr_vec2 kr_cvec2;
typedef const kr_vec3 kr_cvec3;
typedef const kr_vec4 kr_cvec4;

typedef union  { struct { kr_scalar lb[3]; kr_scalar ub[3]; }; struct { vec3 min; vec3 max; }; kr_vec3 bounds[2]; } aabb3, kr_aabb3;
typedef struct { kr_vec3 mid, v0, v1, v2, ext; } kr_obb3;

typedef union { kr_scalar v[16]; struct { vec4 cols[4]; }; kr_scalar c[4][4]; struct { kr_scalar c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33; }; } mat4, kr_mat4;
typedef union { kr_scalar v[12];  struct { vec3 cols[4]; }; kr_scalar c[4][3]; struct { kr_scalar c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32; }; } mat43, kr_mat43;
typedef union { kr_scalar v[9];  struct { vec3 cols[3]; }; kr_scalar c[3][3]; struct { kr_scalar c00, c01, c02, c10, c11, c12, c20, c21, c22; }; } mat3, kr_mat3;

typedef union { f32 v[2]; struct { f32 x, y; }; struct { f32 r, g; }; } fvec2, kr_fvec2;
typedef union { f32 v[3]; struct { f32 x, y, z; }; struct { f32 r, g, b; }; } fvec3, kr_fvec3;
typedef union { f32 v[4]; struct { f32 x, y, z, w; }; struct { f32 r, g, b, a; }; } fvec4, kr_fvec4;

typedef union { f64 v[2]; struct { f64 x, y; }; struct { f64 r, g; }; } dvec2, kr_dvec2;
typedef union { f64 v[3]; struct { f64 x, y, z; }; struct { f64 r, g, b; }; } dvec3, kr_dvec3;
typedef union { f64 v[4]; struct { f64 x, y, z, w; }; struct { f64 r, g, b, a; }; } dvec4, kr_dvec4;

typedef union { u32 v[2]; struct { u32 x, y; }; struct { u32 r, g; }; } uvec2, kr_uvec2;
typedef union { u32 v[3]; struct { u32 x, y, z; }; struct { u32 r, g, b; }; } uvec3, kr_uvec3;
typedef union { u32 v[4]; struct { u32 x, y, z, w; }; struct { u32 r, g, b, a; }; } uvec4, kr_uvec4;

typedef const kr_uvec2 cuvec2;
typedef const kr_uvec3 cuvec3;
typedef const kr_uvec4 cuvec4;

typedef const kr_mat4  cmat4;

typedef union { b32 v[3]; struct { b32 x, y, z; }; struct { b32 r, g, b; }; } bvec3, kr_bvec3;

typedef union { u8 v[4]; struct { u8 x, y, z, w; }; struct { u8 r, g, b, a; }; } u8vec4, kr_u8vec4;
typedef union { u8 v[3]; struct { u8 x, y, z; }; struct { u8 r, g, b; }; } u8vec3, kr_u8vec3;
typedef union { u8 v[2]; struct { u8 x, y; }; struct { u8 r, g; }; } u8vec2, kr_u8vec;

typedef union { i32 v[2]; struct { i32 x, y; }; struct { i32 r, g; }; } ivec2, kr_ivec2;
typedef union { i32 v[3]; struct { i32 x, y, z; }; struct { i32 r, g, b; }; } ivec3, kr_ivec3;
typedef union { i32 v[4]; struct { i32 x, y, z, w; }; struct { i32 r, g, b, a; }; } ivec4, kr_ivec4;

typedef const kr_ivec2 civec2;
typedef const kr_ivec3 civec3;
typedef const kr_ivec4 civec4;

typedef struct
{
    kr_vec3 tangent;
    kr_vec3 bitangent;
    kr_vec3 normal;
} kr_onb;

typedef struct {
    kr_mat4 to;
    kr_mat4 from;
} kr_transform;

#define KR_PI_F 3.14159265359f
#define KR_PI_D 3.14159265359
#define KR_PI ((kr_scalar)KR_PI_D)
#define KR_PI_OVER_4_F 0.78539816339744830961f
#define KR_PI_OVER_4_D 0.78539816339744830961
#define KR_PI_OVER_4 ((kr_scalar)KR_PI_OVER_4_D)
#define KR_PI_OVER_2_F 1.57079632679489661923f
#define KR_PI_OVER_2_D 1.57079632679489661923
#define KR_PI_OVER_2 ((kr_scalar)KR_PI_OVER_2_D)
/*
static PBRT_CONSTEXPR Float ShadowEpsilon = 0.0001f;
static PBRT_CONSTEXPR Float Pi = 3.14159265358979323846;
static PBRT_CONSTEXPR Float InvPi = 0.31830988618379067154;
static PBRT_CONSTEXPR Float Inv2Pi = 0.15915494309189533577;
static PBRT_CONSTEXPR Float Inv4Pi = 0.07957747154594766788;
static PBRT_CONSTEXPR Float PiOver2 = 1.57079632679489661923;
static PBRT_CONSTEXPR Float PiOver4 = 0.78539816339744830961;
static PBRT_CONSTEXPR Float Sqrt2 = 1.41421356237309504880;
*/

#ifdef __cplusplus
//extern "C" {
#endif

    kr_inline_host_device kr_vec2 kr_ray_aabb_intersect_n(kr_vec3 rd, kr_vec3 ro, kr_aabb3 bbox, kr_scalar t_max, kr_vec3* normal);
    kr_inline_host_device kr_vec2 kr_ray_aabb_intersect(kr_vec3 rd, kr_vec3 ro, kr_aabb3 bbox, kr_scalar t_max);
    kr_inline_host_device kr_vec2 kr_ray_unit_aabb_intersect(kr_vec3 rd, kr_vec3 ro, kr_scalar t_max);
    kr_inline_host_device kr_vec2 kr_ray_unit_aabb_intersect_n(kr_vec3 rd, kr_vec3 ro, kr_scalar t_max, kr_vec3* normal);
    kr_inline_host_device vec3 kr_ray_triangle_intersect(vec3 direction, vec3 origin, vec3 v1, vec3 v2, vec3 v3, kr_scalar t_max);

    kr_inline_host_device kr_u32    kr_ctz32(kr_u32 value);
    kr_inline_host_device kr_u64    kr_ctz64(kr_u64 value);
    kr_inline_host_device kr_u32    kr_clz32(kr_u32 value);
    kr_inline_host_device kr_u64    kr_clz64(kr_u64 value);
    kr_inline_host_device kr_scalar kr_min(kr_scalar a, kr_scalar b);
    kr_inline_host_device kr_scalar kr_max(kr_scalar a, kr_scalar b);
    kr_inline_host_device kr_f64 kr_minf64(kr_f64 a, kr_f64 b);
    kr_inline_host_device kr_f64 kr_maxf64(kr_f64 a, kr_f64 b);
    kr_inline_host_device kr_i32 kr_mini(kr_i32 a, kr_i32 b);
    kr_inline_host_device kr_i32 kr_maxi(kr_i32 a, kr_i32 b);
    kr_inline_host_device kr_u32 kr_maxu(kr_u32 a, kr_u32 b);
    kr_inline_host_device kr_u32 kr_minu(kr_u32 a, kr_u32 b);
    kr_inline_host_device kr_scalar kr_cos(kr_scalar a);
    kr_inline_host_device kr_scalar kr_sin(kr_scalar a);
    kr_inline_host_device kr_scalar kr_sqrt(kr_scalar a);
    kr_inline_host_device kr_scalar kr_abs(kr_scalar a);
    kr_inline_host_device kr_b32    kr_isinf(kr_scalar a);

    kr_inline_host_device kr_scalar kr_min3(kr_vec3 a);
    kr_inline_host_device kr_scalar kr_max3(kr_vec3 a);

    kr_inline_host_device kr_bvec3 kr_vepsilon_equal3(kr_vec3 a, kr_vec3 b, kr_scalar eps);

    kr_inline_host_device kr_b32 kr_vany3(kr_bvec3 a);

    kr_inline_host_device kr_vec3 kr_vsign3(kr_vec3 a);
    kr_inline_host_device vec3 kr_vmin3(vec3 a, vec3 b);
    kr_inline_host_device vec3 kr_vmax3(vec3 a, vec3 b);
    kr_inline_host_device vec3 kr_vsafeinv3(vec3 a);
    kr_inline_host_device vec3 kr_vinv3(vec3 a);
    kr_inline_host_device kr_uvec2 kr_uvclamp2(kr_uvec2 a, kr_uvec2 lb, kr_uvec2 ub);
    kr_inline_host_device kr_uvec3 kr_uvclamp3(kr_uvec3 a, kr_uvec3 lb, kr_uvec3 ub);
    kr_inline_host_device kr_scalar kr_clamp(kr_scalar a, kr_scalar lb, kr_scalar ub);
    kr_inline_host_device kr_i32    kr_clampi(kr_i32 a, kr_i32 lb, kr_i32 ub);
    kr_inline_host_device kr_u32    kr_clampu(kr_u32 a, kr_u32 lb, kr_u32 ub);
    kr_inline_host_device kr_u32    kr_bitcount(kr_u32 i);
    kr_inline_host_device kr_u8vec4 kr_u8vmix4(kr_u8vec4 x, kr_u8vec4 y, kr_scalar a);
    kr_inline_host_device kr_u8vec3 kr_u8vmix3(kr_u8vec3 x, kr_u8vec3 y, kr_scalar a);
    kr_inline_host_device kr_vec3 kr_vstep3(kr_vec3 edge, kr_vec3 v);
    kr_inline_host_device kr_vec3 kr_vmix3(vec3 x, vec3 y, kr_scalar a);
    kr_inline_host_device kr_vec4 kr_vmix4(vec4 x, vec4 y, kr_scalar a);
    kr_inline_host_device kr_vec2 kr_vfract2(vec2 a);
    kr_inline_host_device kr_scalar kr_fract(kr_scalar a);
    kr_inline_host_device kr_scalar kr_floor(kr_scalar a);
    kr_inline_host_device kr_u8vec3 kr_vrgb3(vec3 a);
    kr_inline_host_device kr_u8vec4 kr_vrgba2(kr_vec2 a);
    kr_inline_host_device kr_u8vec4 kr_vrgba3(vec3 a);
    kr_inline_host_device kr_u8vec4 kr_vrgba4(vec4 a);
    kr_inline_host_device kr_vec3 kr_u8vnorm3(kr_u8vec3 a);
    kr_inline_host_device kr_vec4 kr_u8vnorm4(kr_u8vec4 a);
    kr_inline_host_device kr_vec3 kr_v3dinterpolate3(vec3 a, vec3 b, vec3 c, vec3 barys);
    kr_inline_host_device kr_vec2 kr_v3dinterpolate2(vec2 a, vec2 b, vec2 c, vec3 barys);
    kr_inline_host_device kr_vec3 kr_vpow3(kr_vec3 a, kr_scalar p);

    kr_inline_host_device b32     kr_vequal3(vec3 a, vec3 b);
    kr_inline_host_device kr_vec2 kr_vzero2();
    kr_inline_host_device kr_vec3 kr_vzero3();
    kr_inline_host_device kr_vec3 kr_vof3(kr_scalar a);
    kr_inline_host_device kr_vec4 kr_vzero4();
    kr_inline_host_device kr_scalar kr_vlength3sqr(vec3 a);
    kr_inline_host_device kr_scalar kr_vlength3(vec3 a);
    kr_inline_host_device kr_scalar kr_vdistance_to_inf_edge3(kr_vec3 q, kr_vec3 p0, kr_vec3 v);
    kr_inline_host_device kr_scalar kr_vdistance3(kr_vec3 a, kr_vec3 b);
    kr_inline_host_device kr_scalar kr_vdistance3sqr(kr_vec3 a, kr_vec3 b);
    kr_inline_host_device vec3 kr_vnegate3(vec3 a);
    kr_inline_host_device kr_vec3 kr_vnormalize3(vec3 a);
    kr_inline_host_device kr_uvec3 kr_vu3clamp1(kr_uvec3 v, kr_u32 l, kr_u32 u);
    kr_inline_host_device kr_ivec3 kr_vi3clamp1(kr_ivec3 v, kr_i32 l, kr_i32 u);
    kr_inline_host_device kr_vec3 kr_vclamp31(kr_vec3 v, kr_scalar l, kr_scalar u);
    kr_inline_host_device kr_vec3 kr_vclamp3(kr_vec3 a, kr_vec3 l, kr_vec3 u);
    kr_inline_host_device kr_vec3 kr_vabs3(kr_vec3 a);
    kr_inline_host_device kr_vec3 kr_vto43(kr_vec4 a);
    kr_inline_host_device kr_bvec3 kr_vgeq3(kr_vec3 a, kr_vec3 b);
    kr_inline_host_device kr_bvec3 kr_vle3(kr_vec3 a, kr_vec3 b);
    kr_inline_host_device u32 kr_vpack3b(kr_bvec3 a);
    kr_inline_host_device kr_scalar kr_vdot3(vec3 a, vec3 b);
    kr_inline_host_device vec3  kr_vflatten3(vec3 a, vec3 b, kr_scalar eps);
    kr_inline_host_device vec3  kr_vinverse3(vec3 a);
    kr_inline_host_device vec3  kr_vdiv31(vec3 a, kr_scalar b);
    kr_inline_host_device vec3  kr_vmul3(vec3 a, vec3 b);
    kr_inline_host_device vec3  kr_vadd3(vec3 a, vec3 b);
    kr_inline_host_device vec3  kr_vadd31(vec3 a, kr_scalar b);
    kr_inline_host_device vec2  kr_vadd2(vec2 a, vec2 b);
    kr_inline_host_device kr_uvec3  kr_vadd3u(kr_uvec3 a, kr_uvec3 b);
    kr_inline_host_device kr_uvec4  kr_vadd4u(kr_uvec4 a, kr_uvec4 b);
    kr_inline_host_device kr_ivec3  kr_viadd3(kr_ivec3 a, kr_ivec3 b);
    kr_inline_host_device kr_ivec4  kr_viadd4(kr_ivec4 a, kr_ivec4 b);
    kr_inline_host_device vec3  kr_vsub3(vec3 a, vec3 b);
    kr_inline_host_device vec2  kr_vsub2(vec2 a, vec2 b);
    kr_inline_host_device vec3  kr_vdiv3(vec3 a, vec3 b);
    kr_inline_host_device vec3  kr_vdiv3s(vec3 a, vec3 b);
    kr_inline_host_device vec3  kr_vmul31(vec3 a, kr_scalar b);
    kr_inline_host_device vec2  kr_vmul21(vec2 a, kr_scalar b);
    kr_inline_host_device vec3  kr_vcross3(vec3 a, vec3 b);
    kr_inline_host_device kr_mat4 kr_mquat4(kr_vec4 quat);
    kr_inline_host_device kr_mat4 kr_mlookat4(kr_vec3 eye, kr_vec3 center, kr_vec3 up);
    kr_inline_host_device kr_mat4  kr_mscreen4(u32 width, u32 height);
    kr_inline_host_device kr_mat4  kr_mfrom16(kr_scalar m00, kr_scalar m01, kr_scalar m02, kr_scalar m03, kr_scalar m10, kr_scalar m11, kr_scalar m12, kr_scalar m13, kr_scalar m20, kr_scalar m21, kr_scalar m22, kr_scalar m23, kr_scalar m30, kr_scalar m31, kr_scalar m32, kr_scalar m33);
    kr_inline_host_device kr_mat43  kr_m43from4(kr_mat4);
    kr_inline_host_device kr_mat4  kr_midentity4();
    kr_inline_host_device kr_mat4  kr_mperspective4(kr_scalar fov_y, kr_scalar aspect, kr_scalar znear, kr_scalar zfar);
    kr_inline_host_device kr_mat4  kr_morthographic4(kr_scalar left, kr_scalar right, kr_scalar bottom, kr_scalar top, kr_scalar znear, kr_scalar zfar);
    kr_inline_host_device kr_mat4  kr_mtranspose4(kr_mat4 m);
    kr_inline_host_device kr_mat4  kr_minverse4(kr_mat4 m);
    kr_inline_host_device kr_mat4  kr_mscale4(vec3 scale);
    kr_inline_host_device kr_mat4  kr_mtranslate4(vec3 move);
    kr_inline_host_device kr_transform  kr_mtransform4(kr_mat4 m);
    kr_inline_host_device kr_transform  kr_minvtransform4(kr_mat4 m);
    kr_inline_host_device kr_transform  kr_invtransform(kr_transform m);
    kr_inline_host_device kr_mat4  kr_mrotate4(vec3 axis, kr_scalar radians);
    kr_inline_host_device kr_vec4  kr_mangle_axis4(kr_mat4 m);
    kr_inline_host_device kr_vec3  kr_vrotate_angle_axis3(kr_vec3 v, kr_vec3 axis, kr_scalar radians);
    
    kr_inline_host_device kr_mat4  kr_mrows3(kr_vec3 x, kr_vec3 y, kr_vec3 z, kr_vec3 w);
    kr_inline_host_device kr_mat4  kr_mobb3(kr_obb3 obb);
    kr_inline_host_device void  kr_minvert4(mat4* m);
    kr_inline_host_device mat4  kr_mmul4(mat4 l, mat4 r);
    kr_inline_host_device vec3  kr_mvmul43(mat4 l, vec3 r);
    kr_inline_host_device vec4  kr_mvmul44(mat4 m, vec4 v);
    kr_inline_host_device vec3  kr_vtransform3p(const mat4* l, const vec3* r);
    kr_inline_host_device vec3  kr_v43transform3p(const mat43* l, const vec3* r);
    kr_inline_host_device vec3  kr_vtransform3(mat4 l, vec3 r);
    kr_inline_host_device vec4  kr_vtransform4(mat4 l, vec4 r);
    kr_inline_host_device vec3  kr_ntransform3(mat4 l, vec3 r);
    kr_inline_host_device vec3  kr_ntransform3p(const mat4* l, const vec3* r);
    kr_inline_host_device vec3  kr_n43transform3p(const mat43* l, const vec3* r);
    kr_inline_host_device vec3  kr_vproject3(mat4 l, vec3 r);
    kr_inline_host_device vec3  kr_voffset3(vec3 o, vec3 d, kr_scalar s);
    kr_inline_host_device kr_u32  kr_v32morton3(vec3 a);
    kr_inline_host_device kr_u64  kr_v64morton3(vec3 a);
    kr_inline_host_device kr_u32 kr_reverseu32(kr_u32 x);

    kr_inline_host_device vec3  kr_obb_extents3(kr_obb3 obb);
    kr_inline_host_device kr_scalar kr_obb_surface_area3(kr_obb3 obb);
    kr_inline_host_device kr_scalar kr_aabb_surface_area3(aabb3 box);
    kr_inline_host_device kr_scalar kr_aabb_volume3(aabb3 box);
    kr_inline_host_device vec3  kr_aabb_center3(aabb3 box);
    kr_inline_host_device kr_scalar  kr_aabb_radius3(aabb3 box);
    kr_inline_host_device kr_aabb3 kr_aabb_empty3();
    kr_inline_host_device vec3  kr_aabb_extents3(aabb3 box);
    kr_inline_host_device aabb3 kr_aabb_shrink3(aabb3 box, kr_scalar v);
    kr_inline_host_device aabb3 kr_aabb_blow3(aabb3 box, kr_scalar v);
    kr_inline_host_device void  kr_aabb_corners3(kr_aabb3 box, kr_vec3* points);
    kr_inline_host_device void  kr_aabb_split3(aabb3 box, aabb3* children);
    kr_inline_host_device aabb3 kr_aabb_create3(vec3 vmin, vec3 vmax);
    kr_inline_host_device aabb3 kr_aabb_offset3(aabb3 box, vec3 a);
    kr_inline_host_device aabb3 kr_aabb_expand3(aabb3 box, vec3 a);
    kr_inline_host_device aabb3 kr_aabb_expand(aabb3 box, aabb3 a);
    kr_inline_host_device aabb3 kr_aabb_transform4(mat4 m, aabb3 box);

    kr_inline_host_device kr_scalar kr_copysign(kr_scalar a, kr_scalar b);
    kr_inline_host_device kr_scalar kr_sign(kr_scalar a);
    kr_inline_host_device kr_i32 kr_signi(kr_i32 a);

    kr_inline_host_device kr_scalar kr_lerp(kr_scalar x, kr_scalar y, kr_scalar a);
    kr_inline_host_device kr_vec4 kr_vlerp4(kr_vec4 x, kr_vec4 y, kr_scalar a);
    kr_inline_host_device kr_vec3 kr_vlerp3(kr_vec3 x, kr_vec3 y, kr_scalar a);

    kr_inline_host_device kr_scalar kr_radians(kr_scalar dgrs);
    kr_inline_host_device kr_scalar kr_degrees(kr_scalar rads);

    kr_inline_host_device void kr_onb_create(kr_vec3 normal, kr_onb* onb);
    kr_inline_host_device kr_vec3 kr_onb_vto3(kr_onb* onb, kr_vec3 v);
    kr_inline_host_device kr_vec3 kr_onb_vfrom3(kr_onb* onb, kr_vec3 v);
    kr_inline_host_device kr_scalar kr_quantize(kr_scalar a);
    kr_inline_host_device kr_vec3 kr_vquantize3(kr_vec3 v);

#ifdef KR_DEBUG_FACILITIES
#define kr_vprint3(a) kr__vprint3(#a, a)
	void kr__vprint3(const char* name, kr_vec3 a);
#define kr_aabb_print3(a) kr__aabb_print3(#a, a)
    void kr__aabb_print3(const char* name, kr_aabb3 a);
#define kr_bit_printu32(a) kr__bit_printu32(#a, a)
    void kr__bit_printu32(const char* name, u32 a);
#define kr_bit_printu64(a) kr__bit_printu64(#a, a)
    void kr__bit_printu64(const char* name, u64 a);
#define kr_bit_fprintu32(f,a) kr__bit_fprintu32(f,#a, a)
    void kr__bit_fprintu32(void* f, const char* name, u32 a);
#endif /* KR_DEBUG_FACILITIES */

#ifdef __cplusplus
//}
#endif

#if defined( KR_VECMATH_IMPL )

    kr_inline_host_device kr_u8vec3 kr_vrgb3(vec3 a) {
        return KR_INITIALIZER_CAST(kr_u8vec3) {
            (kr_u8)(a.x * 255.0f),
            (kr_u8)(a.y * 255.0f),
            (kr_u8)(a.z * 255.0f)
        };
    }

    kr_inline_host_device kr_u8vec4 kr_vrgba2(kr_vec2 a) {
        return KR_INITIALIZER_CAST(kr_u8vec4) {
            (kr_u8)(a.x * 255.0f),
            (kr_u8)(a.y * 255.0f),
            0,
            0
        };
    }

    kr_inline_host_device kr_u8vec4 kr_vrgba3(vec3 a) {
        return KR_INITIALIZER_CAST(kr_u8vec4) {
            (kr_u8)(a.x * 255.0f),
            (kr_u8)(a.y * 255.0f),
            (kr_u8)(a.z * 255.0f),
             0
        };
    }

    kr_inline_host_device kr_u8vec4 kr_vrgba4(vec4 a) {
        return KR_INITIALIZER_CAST(kr_u8vec4) {
            (kr_u8)(a.x * 255.0f),
            (kr_u8)(a.y * 255.0f),
            (kr_u8)(a.z * 255.0f),
            (kr_u8)(a.w * 255.0f),
        };
    }

kr_inline_host_device kr_vec3 kr_u8vnorm3(kr_u8vec3 a) {
    return KR_INITIALIZER_CAST(kr_vec3) {
      ((kr_scalar)a.x / 255.0f),
      ((kr_scalar)a.y / 255.0f),
      ((kr_scalar)a.z / 255.0f)
    };
}

kr_inline_host_device kr_vec4 kr_u8vnorm4(kr_u8vec4 a) {
    return KR_INITIALIZER_CAST(kr_vec4) {
      ((kr_scalar)a.x / 255.0f),
      ((kr_scalar)a.y / 255.0f),
      ((kr_scalar)a.z / 255.0f),
      ((kr_scalar)a.w / 255.0f),
    };
}

kr_inline_host_device kr_vec3 kr_v3dinterpolate3(vec3 a, vec3 b, vec3 c, vec3 barys) {
    cvec3 v0 = kr_vmul31(a, barys.x);
    cvec3 v1 = kr_vmul31(b, barys.y);
    cvec3 v2 = kr_vmul31(c, barys.z);
    return kr_vadd3(kr_vadd3(v0, v1), v2);
}

kr_inline_host_device kr_vec2 kr_v3dinterpolate2(vec2 a, vec2 b, vec2 c, vec3 barys) {
    cvec2 v0 = kr_vmul21(a, barys.x);
    cvec2 v1 = kr_vmul21(b, barys.y);
    cvec2 v2 = kr_vmul21(c, barys.z);
    return kr_vadd2(kr_vadd2(v0, v1), v2);
}

kr_inline_host_device kr_vec3 kr_vpow3(kr_vec3 a, kr_scalar p) {
    return KR_INITIALIZER_CAST(kr_vec3) { 
        (kr_scalar)pow(a.x, p),
        (kr_scalar)pow(a.y, p),
        (kr_scalar)pow(a.z, p)
    };
}

kr_inline_host_device b32 kr_vequal3(vec3 a, vec3 b) {
    return (
        kr_abs(a.x - b.x) < 0.0001 &&
        kr_abs(a.y - b.y) < 0.0001 &&
        kr_abs(a.z - b.z) < 0.0001
    );
}

kr_inline_host_device kr_vec2 kr_vzero2() {
    return KR_INITIALIZER_CAST(kr_vec2) { 0, 0 };
}

kr_inline_host_device kr_vec3 kr_vzero3() {
    return KR_INITIALIZER_CAST(kr_vec3) { 0, 0, 0 };
}

kr_inline_host_device kr_vec3 kr_vof3(kr_scalar a) {
    return KR_INITIALIZER_CAST(kr_vec3) { a, a, a };
}

kr_inline_host_device kr_vec4 kr_vzero4() {
    return KR_INITIALIZER_CAST(kr_vec4) { 0, 0, 0, 0 };
}

kr_inline_host_device kr_scalar kr_vdistance3sqr(kr_vec3 a, kr_vec3 b) {
    return kr_vlength3sqr(kr_vsub3(b, a));
}

kr_inline_host_device kr_scalar
kr_vdistance_to_inf_edge3(kr_vec3 q, kr_vec3 p0, kr_vec3 v) {
    kr_vec3 u0 = kr_vsub3(q, p0);
    kr_scalar t = kr_vdot3(v, u0);
    kr_scalar sqLen_v = kr_vlength3sqr(v);
    return kr_vlength3sqr(u0) - t * t / sqLen_v;
}

kr_inline_host_device kr_scalar kr_vdistance3(kr_vec3 a, kr_vec3 b) {
    return (kr_scalar)sqrt(kr_vdistance3sqr(a, b));
}

kr_inline_host_device kr_scalar kr_vlength3sqr(vec3 a) {
    return (kr_scalar)(a.x * a.x + a.y * a.y + a.z * a.z);
}

kr_inline_host_device kr_scalar kr_vlength3(vec3 a) {
    return (kr_scalar)sqrt(kr_vlength3sqr(a));
}

kr_inline_host_device vec3 kr_vnegate3(vec3 a) {
    return KR_INITIALIZER_CAST(vec3) { -a.x, -a.y, -a.z };
}

kr_inline_host_device vec3 kr_vnormalize3(vec3 a) {
    return kr_vdiv31(a, kr_vlength3(a));
}

kr_inline_host_device kr_vec3 kr_vto43(kr_vec4 a) {
    return KR_INITIALIZER_CAST(vec3) { a.x, a.y, a.z };
}

kr_inline_host_device u32 kr_vpack3b(kr_bvec3 a) {
    return ( ( a.x << 2 ) | (a.y << 1) | (a.z << 0) );
}

kr_inline_host_device kr_bvec3 kr_vgeq3(kr_vec3 a, kr_vec3 b) {
    return KR_INITIALIZER_CAST(kr_bvec3) { (a.x >= b.x) ? 1 : 0, (a.y >= b.y) ? 1 : 0, (a.z >= b.z) ? 1 : 0 };
}

kr_inline_host_device kr_bvec3 kr_vle3(kr_vec3 a, kr_vec3 b) {
    return KR_INITIALIZER_CAST(kr_bvec3) { (a.x < b.x) ? 1 : 0, (a.y < b.y) ? 1 : 0, (a.z < b.z) ? 1 : 0 };
}

kr_inline_host_device kr_vec3 kr_vclamp3(kr_vec3 v, kr_vec3 l, kr_vec3 u) {
    return KR_INITIALIZER_CAST(vec3) {
        kr_min(kr_max(v.x, l.x), u.x),
        kr_min(kr_max(v.y, l.y), u.y),
        kr_min(kr_max(v.z, l.z), u.z)
    };
}

kr_inline_host_device kr_vec3 kr_vclamp31(kr_vec3 v, kr_scalar l, kr_scalar u) {
    return KR_INITIALIZER_CAST(vec3) {
        kr_min(kr_max(v.x, l), u),
        kr_min(kr_max(v.y, l), u),
        kr_min(kr_max(v.z, l), u)
    };
}

kr_inline_host_device kr_uvec3 kr_vu3clamp1(kr_uvec3 v, kr_u32 l, kr_u32 u) {
    return KR_INITIALIZER_CAST(uvec3) {
        kr_minu(kr_maxu(v.x, l), u),
        kr_minu(kr_maxu(v.y, l), u),
        kr_minu(kr_maxu(v.z, l), u)
    };
}

kr_inline_host_device kr_ivec3 kr_vi3clamp1(kr_ivec3 v, kr_i32 l, kr_i32 u) {
    return KR_INITIALIZER_CAST(ivec3) {
        kr_mini(kr_maxi(v.x, l), u),
        kr_mini(kr_maxi(v.y, l), u),
        kr_mini(kr_maxi(v.z, l), u)
    };
}

kr_inline_host_device kr_vec3 kr_vabs3(kr_vec3 a) {
    return KR_INITIALIZER_CAST(vec3) {
        (a.x < 0.0f) ? -a.x : a.x,
        (a.y < 0.0f) ? -a.y : a.y,
        (a.z < 0.0f) ? -a.z : a.z,
    };
}

kr_inline_host_device kr_scalar kr_clamp(kr_scalar a, kr_scalar lb, kr_scalar ub) {
    return (a < lb) ? lb : (ub < a) ? ub : a;
}
kr_inline_host_device kr_i32 kr_clampi(kr_i32 a, kr_i32 lb, kr_i32 ub) {
    return (a < lb) ? lb : (ub < a) ? ub : a;
}
kr_inline_host_device kr_u32 kr_clampu(kr_u32 a, kr_u32 lb, kr_u32 ub) {
    return (a < lb) ? lb : (ub < a) ? ub : a;
}
kr_inline_host_device kr_uvec2 kr_uvclamp2(kr_uvec2 a, kr_uvec2 lb, kr_uvec2 ub) {
    return KR_INITIALIZER_CAST(kr_uvec2) {
        kr_clampu(a.x, lb.x, ub.x),
        kr_clampu(a.y, lb.y, ub.y)
    };
}
kr_inline_host_device kr_uvec3 kr_uvclamp3(kr_uvec3 a, kr_uvec3 lb, kr_uvec3 ub) {
    return KR_INITIALIZER_CAST(kr_uvec3) {
        kr_clampu(a.x, lb.x, ub.x),
        kr_clampu(a.y, lb.y, ub.y),
        kr_clampu(a.z, lb.z, ub.z)
    };
}

kr_inline_host_device kr_vec3 kr_vstep3(kr_vec3 edge, kr_vec3 v) {
    return KR_INITIALIZER_CAST(kr_vec3) {
        v.x < edge.x ? (kr_scalar)0 : (kr_scalar)1,
        v.y < edge.y ? (kr_scalar)0 : (kr_scalar)1,
        v.z < edge.z ? (kr_scalar)0 : (kr_scalar)1
    };
}


kr_inline_host_device kr_vec3 kr_vmix3(vec3 x, vec3 y, kr_scalar a) {
    return KR_INITIALIZER_CAST(kr_vec3) {
        x.x * (1.0f - a) + a * y.x,
        x.y * (1.0f - a) + a * y.y,
        x.z * (1.0f - a) + a * y.z
    };
}

kr_inline_host_device kr_u8vec4 kr_u8vmix4(kr_u8vec4 x, kr_u8vec4 y, kr_scalar a) {
    return kr_vrgba4(
        kr_vmix4(
            kr_u8vnorm4(x), kr_u8vnorm4(y), a)
        );
}

kr_inline_host_device kr_u8vec3 kr_u8vmix3(kr_u8vec3 x, kr_u8vec3 y, kr_scalar a) {
    return kr_vrgb3(
        kr_vmix3(
            kr_u8vnorm3(x), kr_u8vnorm3(y), a)
        );
}


kr_inline_host_device kr_vec4 
kr_vmix4(vec4 x, vec4 y, kr_scalar a) {
    return KR_INITIALIZER_CAST(kr_vec4) {
        x.x * (1.0f - a) + a * y.x,
        x.y * (1.0f - a) + a * y.y,
        x.z * (1.0f - a) + a * y.z,
        x.w * (1.0f - a) + a * y.w
    };
}

kr_inline_host_device kr_scalar kr_floor(kr_scalar a) {
    if (a == 0.0) return 0.0f;
    else if (a > 0) return (kr_scalar)((long)a);
    else return (kr_scalar)(((long)a) - 1);
}

kr_inline_host_device kr_scalar kr_fract(kr_scalar a) {
    return a - kr_floor(a);
}

kr_inline_host_device kr_vec2 kr_vfract2(vec2 a) {
    return KR_INITIALIZER_CAST(kr_vec2) {
        kr_fract(a.x),
        kr_fract(a.y),
    };
}

kr_inline_host_device kr_u32 kr_bitcount(kr_u32 i) {
    i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  // quads
    i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
    return (i * 0x01010101) >> 24;          // horizontal sum of bytes
}

//https://github.com/mmp/pbrt-v3/blob/master/src/accelerators/bvh.cpp
kr_inline_host_device
kr_u32 kr_32strech_by_3(kr_u32 x)
{
    x = x & 0x3ffu;
    x = (x | (x << 16)) & 0x30000ff;
    x = (x | (x << 8)) & 0x300f00f;
    x = (x | (x << 4)) & 0x30c30c3;
    x = (x | (x << 2)) & 0x9249249;

    return x;
}

kr_inline_host_device kr_u32  kr_v32morton3(vec3 a) {
    return (
        kr_32strech_by_3((u32)kr_min(kr_max(a.x * 1024.0f, 0.0f), 1023.0f)) << 2 |
        kr_32strech_by_3((u32)kr_min(kr_max(a.y * 1024.0f, 0.0f), 1023.0f)) << 1 |
        kr_32strech_by_3((u32)kr_min(kr_max(a.z * 1024.0f, 0.0f), 1023.0f)) << 0
        );
}

kr_inline_host_device kr_u64 kr_64strech_by_3(kr_u64 x)
{
    x = x & 0x1fffffu;
    x = (x | (x << 32)) & 0x1f00000000ffffu;
    x = (x | (x << 16)) & 0x1f0000ff0000ffu;
    x = (x | (x << 8)) & 0x100f00f00f00f00fu;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3u;
    x = (x | (x << 2)) & 0x1249249249249249u;

    return x;
}

kr_inline_host_device kr_u64  kr_v64morton3(vec3 a) {
    return (
        kr_64strech_by_3((kr_u64)kr_min(kr_max(a.x * 2097152.0f, 0.0f), 2097151.0f)) << 2 |
        kr_64strech_by_3((kr_u64)kr_min(kr_max(a.y * 2097152.0f, 0.0f), 2097151.0f)) << 1 |
        kr_64strech_by_3((kr_u64)kr_min(kr_max(a.z * 2097152.0f, 0.0f), 2097151.0f)) << 0
        );
}

kr_inline_host_device kr_u32 kr_reverseu32(kr_u32 x) {
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
	x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
	x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);

	return x;
}

kr_inline_host_device kr_scalar kr_vdot3(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

kr_inline_host_device kr_scalar kr_vdot4(vec4 a, vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

kr_inline_host_device vec3  kr_vdiv3(vec3 a, vec3 b) {
    return KR_INITIALIZER_CAST(vec3) { a.x / b.x, a.y / b.y, a.z / b.z };
}

kr_inline_host_device vec3  kr_vdiv3s(vec3 a, vec3 b) {
    return KR_INITIALIZER_CAST(vec3) { (b.x != (kr_scalar) 0) ? a.x / b.x : (kr_scalar)0, (b.y != (kr_scalar)0) ? a.y / b.y : (kr_scalar)0, (b.z != (kr_scalar)0) ? a.z / b.z : (kr_scalar)0 };
}

kr_inline_host_device vec3  kr_vflatten3(vec3 a, vec3 b, kr_scalar eps) {
    return KR_INITIALIZER_CAST(vec3) { kr_abs(a.x - b.x) < eps ? b.x : a.x, kr_abs(a.y - b.y) < eps ? b.y : a.y, kr_abs(a.z - b.z) < eps ? b.z : a.z};
}

kr_inline_host_device vec3  kr_vinverse3(vec3 a) {
    return KR_INITIALIZER_CAST(vec3) { (kr_scalar)1.0 / a.x, (kr_scalar)1.0 / a.y, (kr_scalar)1.0 / a.z };
}

kr_inline_host_device vec3  kr_vdiv31(vec3 a, kr_scalar b) {
    return KR_INITIALIZER_CAST(vec3) { a.x / b, a.y / b, a.z / b };
}

kr_inline_host_device vec3  kr_vmul3(vec3 a, vec3 b) {
    return KR_INITIALIZER_CAST(vec3) { a.x * b.x, a.y * b.y, a.z * b.z };
}

kr_inline_host_device vec3  kr_vadd3(vec3 a, vec3 b) {
    return KR_INITIALIZER_CAST(vec3) { a.x + b.x, a.y + b.y, a.z + b.z };
}

kr_inline_host_device vec2  kr_vadd2(vec2 a, vec2 b) {
    return KR_INITIALIZER_CAST(vec2) { a.x + b.x, a.y + b.y };
}

kr_inline_host_device kr_uvec3  kr_vadd3u(kr_uvec3 a, kr_uvec3 b) {
    return KR_INITIALIZER_CAST(kr_uvec3) { a.x + b.x, a.y + b.y, a.z + b.z };
}

kr_inline_host_device kr_uvec4  kr_vadd4u(kr_uvec4 a, kr_uvec4 b) {
    return KR_INITIALIZER_CAST(kr_uvec4) { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

kr_inline_host_device kr_ivec3  kr_viadd3(kr_ivec3 a, kr_ivec3 b) {
    return KR_INITIALIZER_CAST(kr_ivec3) { a.x + b.x, a.y + b.y, a.z + b.z };
}

kr_inline_host_device kr_ivec4  kr_viadd4(kr_ivec4 a, kr_ivec4 b) {
    return KR_INITIALIZER_CAST(kr_ivec4) { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

kr_inline_host_device vec3  kr_vsub3(vec3 a, vec3 b) {
    return KR_INITIALIZER_CAST(vec3) { a.x - b.x, a.y - b.y, a.z - b.z };
}

kr_inline_host_device vec2  kr_vsub2(vec2 a, vec2 b) {
    return KR_INITIALIZER_CAST(vec2) { a.x - b.x, a.y - b.y};
}

kr_inline_host_device vec3  kr_vcross3(vec3 a, vec3 b) {
    return KR_INITIALIZER_CAST(vec3) { a.y* b.z - a.z * b.y, a.z* b.x - a.x * b.z, a.x* b.y - a.y * b.x };
}

kr_inline_host_device mat4  kr_mscale4(vec3 scale) {
    return KR_INITIALIZER_CAST(mat4) {
        scale.x, 0.0f, 0.0f, 0.0f,
            0.0f, scale.y, 0.0f, 0.0f,
            0.0f, 0.0f, scale.z, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
    };
}

kr_inline_host_device mat4  kr_mtranslate4(vec3 move) {
    return KR_INITIALIZER_CAST(mat4) {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        move.x, move.y, move.z, 1.0f,
    };
}

kr_inline_host_device kr_transform  kr_mtransform4(kr_mat4 m) {
    return KR_INITIALIZER_CAST(kr_transform) { m, kr_minverse4(m) };
}

kr_inline_host_device kr_transform  kr_minvtransform4(kr_mat4 m) {
    return KR_INITIALIZER_CAST(kr_transform) { kr_minverse4(m), m };
}

kr_inline_host_device kr_transform  kr_invtransform(kr_transform m) {
	return KR_INITIALIZER_CAST(kr_transform) { m.from, m.to };
}


kr_inline_host_device mat4 kr_mperspective4(kr_scalar fov_y, kr_scalar aspect, kr_scalar znear, kr_scalar zfar) {
    kr_scalar f = 1.0f / (kr_scalar)tan(fov_y / 2.0f);

    return KR_INITIALIZER_CAST(mat4) {
        f / aspect, 0.0, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, zfar / (zfar - znear), 1.0f,
        0.0f, 0.0f, -(zfar * znear) / (zfar - znear), 0.0f
    };
}


kr_inline_host_device kr_mat4
kr_mquat4(kr_vec4 q) {
    kr_scalar qxx = q.x * q.x;
    kr_scalar qyy = q.y * q.y;
    kr_scalar qzz = q.z * q.z;
    kr_scalar qxz = q.x * q.z;
    kr_scalar qxy = q.x * q.y;
    kr_scalar qyz = q.y * q.z;
    kr_scalar qwx = q.w * q.x;
    kr_scalar qwy = q.w * q.y;
    kr_scalar qwz = q.w * q.z;

    return KR_INITIALIZER_CAST(mat4) {
        1.0f - 2.0f * (qyy + qzz), 2.0f * (qxy + qwz)       , 2.0f * (qxz - qwy)       , 0.0f,
        2.0f * (qxy - qwz)       , 1.0f - 2.0f * (qxx + qzz), 2.0f * (qyz + qwx)       , 0.0f,
        2.0f * (qxz + qwy)       , 2.0f * (qyz - qwx)       , 1.0f - 2.0f * (qxx + qyy), 0.0f,
        0.0f                     , 0.0f                     , 0.0f                     , 1.0f
    };

    /*return KR_INITIALIZER_CAST(mat4) {
        1.0f - 2.0f * (qyy + qzz), 2.0f * (qxy - qwz), 2.0f * (qxz + qwy), 0.0f,
            2.0f * (qxy + qwz), 1.0f - 2.0f * (qxx + qzz), 2.0f * (qyz - qwx), 0.0f,
            2.0f * (qxz - qwy), 2.0f * (qyz + qwx), 1.0f - 2.0f * (qxx + qyy), 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
    };*/
}

kr_inline_host_device mat4 kr_morthographic4(kr_scalar left, kr_scalar right, kr_scalar bottom, kr_scalar top, kr_scalar znear, kr_scalar zfar) {
    return KR_INITIALIZER_CAST(mat4) {
        2.0f / (right - left), 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f / (zfar - znear), 0.0f,
            -(right + left) / (right - left), -(top + bottom) / (top - bottom), -znear / (zfar - znear), 1.0f
    };
}

kr_inline_host_device kr_mat43  kr_m43from4(kr_mat4 m) {
    return KR_INITIALIZER_CAST(kr_mat43) {
        m.v[0], m.v[1], m.v[2],
        m.v[4], m.v[5], m.v[6],
        m.v[8], m.v[9], m.v[10],
        m.v[12], m.v[13], m.v[14]
    };
}

kr_inline_host_device mat4  kr_midentity4() {
    return KR_INITIALIZER_CAST(mat4) {
        1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
    };
}

kr_inline_host_device kr_mat4  kr_mfrom16(kr_scalar m00, kr_scalar m01, kr_scalar m02, kr_scalar m03, kr_scalar m10, kr_scalar m11, kr_scalar m12, kr_scalar m13, kr_scalar m20, kr_scalar m21, kr_scalar m22, kr_scalar m23, kr_scalar m30, kr_scalar m31, kr_scalar m32, kr_scalar m33) {
    return KR_INITIALIZER_CAST(mat4) {
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33
    };
}

kr_inline_host_device kr_mat4  kr_mtranspose4(kr_mat4 m) {
    kr_mat4 result;
    result.c[0][0] = m.c[0][0];
    result.c[0][1] = m.c[1][0];
    result.c[0][2] = m.c[2][0];
    result.c[0][3] = m.c[3][0];

    result.c[1][0] = m.c[0][1];
    result.c[1][1] = m.c[1][1];
    result.c[1][2] = m.c[2][1];
    result.c[1][3] = m.c[3][1];

    result.c[2][0] = m.c[0][2];
    result.c[2][1] = m.c[1][2];
    result.c[2][2] = m.c[2][2];
    result.c[2][3] = m.c[3][2];

    result.c[3][0] = m.c[0][3];
    result.c[3][1] = m.c[1][3];
    result.c[3][2] = m.c[2][3];
    result.c[3][3] = m.c[3][3];
    return result;
}

kr_inline_host_device kr_mat4  kr_minverse4(kr_mat4 m) {
    kr_mat4 ret;
    kr_mat4 inv;

    ret.v[0] = m.v[5] * m.v[10] * m.v[15] -
        m.v[5] * m.v[11] * m.v[14] -
        m.v[9] * m.v[6] * m.v[15] +
        m.v[9] * m.v[7] * m.v[14] +
        m.v[13] * m.v[6] * m.v[11] -
        m.v[13] * m.v[7] * m.v[10];

    ret.v[4] = -m.v[4] * m.v[10] * m.v[15] +
        m.v[4] * m.v[11] * m.v[14] +
        m.v[8] * m.v[6] * m.v[15] -
        m.v[8] * m.v[7] * m.v[14] -
        m.v[12] * m.v[6] * m.v[11] +
        m.v[12] * m.v[7] * m.v[10];

    ret.v[8] = m.v[4] * m.v[9] * m.v[15] -
        m.v[4] * m.v[11] * m.v[13] -
        m.v[8] * m.v[5] * m.v[15] +
        m.v[8] * m.v[7] * m.v[13] +
        m.v[12] * m.v[5] * m.v[11] -
        m.v[12] * m.v[7] * m.v[9];

    ret.v[12] = -m.v[4] * m.v[9] * m.v[14] +
        m.v[4] * m.v[10] * m.v[13] +
        m.v[8] * m.v[5] * m.v[14] -
        m.v[8] * m.v[6] * m.v[13] -
        m.v[12] * m.v[5] * m.v[10] +
        m.v[12] * m.v[6] * m.v[9];

    ret.v[1] = -m.v[1] * m.v[10] * m.v[15] +
        m.v[1] * m.v[11] * m.v[14] +
        m.v[9] * m.v[2] * m.v[15] -
        m.v[9] * m.v[3] * m.v[14] -
        m.v[13] * m.v[2] * m.v[11] +
        m.v[13] * m.v[3] * m.v[10];

    ret.v[5] = m.v[0] * m.v[10] * m.v[15] -
        m.v[0] * m.v[11] * m.v[14] -
        m.v[8] * m.v[2] * m.v[15] +
        m.v[8] * m.v[3] * m.v[14] +
        m.v[12] * m.v[2] * m.v[11] -
        m.v[12] * m.v[3] * m.v[10];

    ret.v[9] = -m.v[0] * m.v[9] * m.v[15] +
        m.v[0] * m.v[11] * m.v[13] +
        m.v[8] * m.v[1] * m.v[15] -
        m.v[8] * m.v[3] * m.v[13] -
        m.v[12] * m.v[1] * m.v[11] +
        m.v[12] * m.v[3] * m.v[9];

    ret.v[13] = m.v[0] * m.v[9] * m.v[14] -
        m.v[0] * m.v[10] * m.v[13] -
        m.v[8] * m.v[1] * m.v[14] +
        m.v[8] * m.v[2] * m.v[13] +
        m.v[12] * m.v[1] * m.v[10] -
        m.v[12] * m.v[2] * m.v[9];

    ret.v[2] = m.v[1] * m.v[6] * m.v[15] -
        m.v[1] * m.v[7] * m.v[14] -
        m.v[5] * m.v[2] * m.v[15] +
        m.v[5] * m.v[3] * m.v[14] +
        m.v[13] * m.v[2] * m.v[7] -
        m.v[13] * m.v[3] * m.v[6];

    ret.v[6] = -m.v[0] * m.v[6] * m.v[15] +
        m.v[0] * m.v[7] * m.v[14] +
        m.v[4] * m.v[2] * m.v[15] -
        m.v[4] * m.v[3] * m.v[14] -
        m.v[12] * m.v[2] * m.v[7] +
        m.v[12] * m.v[3] * m.v[6];

    ret.v[10] = m.v[0] * m.v[5] * m.v[15] -
        m.v[0] * m.v[7] * m.v[13] -
        m.v[4] * m.v[1] * m.v[15] +
        m.v[4] * m.v[3] * m.v[13] +
        m.v[12] * m.v[1] * m.v[7] -
        m.v[12] * m.v[3] * m.v[5];

    ret.v[14] = -m.v[0] * m.v[5] * m.v[14] +
        m.v[0] * m.v[6] * m.v[13] +
        m.v[4] * m.v[1] * m.v[14] -
        m.v[4] * m.v[2] * m.v[13] -
        m.v[12] * m.v[1] * m.v[6] +
        m.v[12] * m.v[2] * m.v[5];

    ret.v[3] = -m.v[1] * m.v[6] * m.v[11] +
        m.v[1] * m.v[7] * m.v[10] +
        m.v[5] * m.v[2] * m.v[11] -
        m.v[5] * m.v[3] * m.v[10] -
        m.v[9] * m.v[2] * m.v[7] +
        m.v[9] * m.v[3] * m.v[6];

    ret.v[7] = m.v[0] * m.v[6] * m.v[11] -
        m.v[0] * m.v[7] * m.v[10] -
        m.v[4] * m.v[2] * m.v[11] +
        m.v[4] * m.v[3] * m.v[10] +
        m.v[8] * m.v[2] * m.v[7] -
        m.v[8] * m.v[3] * m.v[6];

    ret.v[11] = -m.v[0] * m.v[5] * m.v[11] +
        m.v[0] * m.v[7] * m.v[9] +
        m.v[4] * m.v[1] * m.v[11] -
        m.v[4] * m.v[3] * m.v[9] -
        m.v[8] * m.v[1] * m.v[7] +
        m.v[8] * m.v[3] * m.v[5];

    ret.v[15] = m.v[0] * m.v[5] * m.v[10] -
        m.v[0] * m.v[6] * m.v[9] -
        m.v[4] * m.v[1] * m.v[10] +
        m.v[4] * m.v[2] * m.v[9] +
        m.v[8] * m.v[1] * m.v[6] -
        m.v[8] * m.v[2] * m.v[5];

    kr_scalar det = m.v[0] * ret.v[0] + m.v[1] * ret.v[4] + m.v[2] * ret.v[8] + m.v[3] * ret.v[12];

    if (det == 0) {
        return kr_midentity4();
    }

    det = 1.0f / det;

    for (u32 i = 0; i < 16; i++) {
        inv.v[i] = ret.v[i] * det;
    }

    return inv;
}

kr_inline_host_device void  kr_minvert4(mat4* m) {
    *m = kr_minverse4(*m);
}

kr_inline_host_device kr_mat4  kr_mobb3(kr_obb3 obb) {
    return kr_mrows3(obb.v0, obb.v1, obb.v2, KR_INITIALIZER_CAST(kr_vec3) {0, 0, 0} );
}

kr_inline_host_device kr_mat4 kr_mrows3(kr_vec3 x, kr_vec3 y, kr_vec3 z, kr_vec3 w) {
    return KR_INITIALIZER_CAST(kr_mat4) {
        x.x, x.y, x.z, 0.0f,
            y.x, y.y, y.z, 0.0f,
            z.x, z.y, z.z, 0.0f,
            w.x, w.y, w.z, 1.0f,
    };
}

/* Huge thank you to https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/ */
kr_inline_host_device kr_vec4  kr_mangle_axis4(kr_mat4 m) {
    kr_scalar epsilon = 0.01f; // margin to allow for rounding errors
    kr_scalar epsilon2 = 0.1f;

    // transpose becasue the code at 'euclideanspace.com' uses row-major notation */
    m = kr_mtranspose4(m);

    if ((kr_abs(m.c01 - m.c10) < epsilon)
     && (kr_abs(m.c02 - m.c20) < epsilon)
     && (kr_abs(m.c12 - m.c21) < epsilon)) {
        // singularity found
        // first check for identity matrix which must have +1 for all terms
        //  in leading diagonaland zero in other terms
        if ((kr_abs(m.c01 + m.c10) < epsilon2)
         && (kr_abs(m.c02 + m.c20) < epsilon2)
         && (kr_abs(m.c12 + m.c21) < epsilon2)
         && (kr_abs(m.c00 + m.c11 + m.c22 - 3) < epsilon2)) {
            // this singularity is identity matrix so angle = 0
            return KR_INITIALIZER_CAST(kr_vec4) { 0, 1, 0, 0 };
        }

        // otherwise this singularity is angle = 180
        kr_scalar angle = KR_PI;
        kr_scalar x = 0.0f;
        kr_scalar y = 1.0f;
        kr_scalar z = 0.0f;
        kr_scalar xx = (m.c00 + 1) / 2;
        kr_scalar yy = (m.c11 + 1) / 2;
        kr_scalar zz = (m.c22 + 1) / 2;
        kr_scalar xy = (m.c01 + m.c10) / 4;
        kr_scalar xz = (m.c02 + m.c20) / 4;
        kr_scalar yz = (m.c12 + m.c21) / 4;
        if ((xx > yy) && (xx > zz)) { // m[0][0] is the largest diagonal term
            if (xx < epsilon) {
                x = (kr_scalar)0;
                y = (kr_scalar)0.7071;
                z = (kr_scalar)0.7071;
            }
            else {
                x = kr_sqrt(xx);
                y = (kr_scalar)xy / (kr_scalar)x;
                z = (kr_scalar)xz / (kr_scalar)x;
            }
        }
        else if (yy > zz) { // m[1][1] is the largest diagonal term
            if (yy < epsilon) {
                x = (kr_scalar)0.7071;
                y = (kr_scalar)0;
                z = (kr_scalar)0.7071;
            }
            else {
                y = kr_sqrt(yy);
                x = xy / y;
                z = yz / y;
            }
        }
        else { // m[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon) {
                x = (kr_scalar)0.7071;
                y = (kr_scalar)0.7071;
                z = (kr_scalar)0;
            }
            else {
                z = kr_sqrt(zz);
                x = xz / z;
                y = yz / z;
            }
        }
        return KR_INITIALIZER_CAST(kr_vec4) { x, y, z, angle };
    }
    kr_scalar s = 
    kr_sqrt((m.c21 - m.c12) * (m.c21 - m.c12)
        +   (m.c02 - m.c20) * (m.c02 - m.c20)
        +   (m.c10 - m.c01) * (m.c10 - m.c01));

    kr_scalar angle = (kr_scalar)acos((m.c00 + m.c11 + m.c22 - (kr_scalar)1) / (kr_scalar)2);
    kr_scalar x = (m.c21 - m.c12) / s;
    kr_scalar y = (m.c02 - m.c20) / s;
    kr_scalar z = (m.c10 - m.c01) / s;

    return KR_INITIALIZER_CAST(kr_vec4) { x, y, z, angle };
}

kr_inline_host_device kr_vec3 kr_vrotate_angle_axis3(kr_vec3 v, kr_vec3 axis, kr_scalar radians) {
    kr_scalar cos_theta = kr_cos(radians);
    kr_scalar sin_theta = kr_sin(radians);
    kr_scalar d = kr_vdot3(axis, v); // dot
    kr_vec3   c = kr_vcross3(axis, v); // cross

    return KR_INITIALIZER_CAST(vec3) {
        cos_theta* v.x + sin_theta * c.x + (1.0f - cos_theta) * d * axis.x,
        cos_theta* v.y + sin_theta * c.y + (1.0f - cos_theta) * d * axis.y,
        cos_theta* v.z + sin_theta * c.z + (1.0f - cos_theta) * d * axis.z,
    };
}

kr_inline_host_device mat4  kr_mrotate4(vec3 axis, kr_scalar radians) {
    vec3  v = kr_vnormalize3(axis);
    kr_scalar cost = (kr_scalar)cos(radians);
    kr_scalar sint = (kr_scalar)sin(radians);

    /* https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations */
    /*return KR_INITIALIZER_CAST(mat4) {
        cost + v.x * v.x * (1.0f - cost), v.x* v.y* (1.0f - cost) - v.z * sint, v.x* v.z* (1.0f - cost) + v.y * sint, 0.0f,
        
        v.y* v.x* (1.0f - cost) + v.z * sint, cost + v.y * v.y * (1.0f - cost), v.y* v.z* (1.0f - cost) - v.x * sint, 0.0f,
        
        v.z* v.x* (1.0f - cost) - v.y * sint, v.z* v.y* (1.0f - cost) + v.x * sint, cost + v.z * v.z * (1.0f - cost), 0.0f,
        
        0.0f, 0.0f, 0.0f, 1.0f
    };*/
    return KR_INITIALIZER_CAST(mat4) {
        cost + v.x * v.x * (1.0f - cost)      , v.y* v.x* (1.0f - cost) + v.z * sint , v.z * v.x * (1.0f - cost) - v.y * sint, 0.0f,
        v.x * v.y * (1.0f - cost) - v.z * sint, cost + v.y * v.y * (1.0f - cost)     , v.z * v.y * (1.0f - cost) + v.x * sint, 0.0f,
        v.x * v.z * (1.0f - cost) + v.y * sint, v.y * v.z* (1.0f - cost) - v.x * sint, cost + v.z * v.z * (1.0f - cost)      , 0.0f,
        0.0f                                  , 0.0f                                 , 0.0f                                  , 1.0f
    };
}

kr_inline_host_device mat4  kr_mmul4(mat4 l, mat4 r) {
    return KR_INITIALIZER_CAST(mat4) {
        l.v[0] * r.v[0] + l.v[4] * r.v[1] + l.v[8] * r.v[2] + l.v[12] * r.v[3],
            l.v[1] * r.v[0] + l.v[5] * r.v[1] + l.v[9] * r.v[2] + l.v[13] * r.v[3],
            l.v[2] * r.v[0] + l.v[6] * r.v[1] + l.v[10] * r.v[2] + l.v[14] * r.v[3],
            l.v[3] * r.v[0] + l.v[7] * r.v[1] + l.v[11] * r.v[2] + l.v[15] * r.v[3],

            l.v[0] * r.v[4] + l.v[4] * r.v[5] + l.v[8] * r.v[6] + l.v[12] * r.v[7],
            l.v[1] * r.v[4] + l.v[5] * r.v[5] + l.v[9] * r.v[6] + l.v[13] * r.v[7],
            l.v[2] * r.v[4] + l.v[6] * r.v[5] + l.v[10] * r.v[6] + l.v[14] * r.v[7],
            l.v[3] * r.v[4] + l.v[7] * r.v[5] + l.v[11] * r.v[6] + l.v[15] * r.v[7],

            l.v[0] * r.v[8] + l.v[4] * r.v[9] + l.v[8] * r.v[10] + l.v[12] * r.v[11],
            l.v[1] * r.v[8] + l.v[5] * r.v[9] + l.v[9] * r.v[10] + l.v[13] * r.v[11],
            l.v[2] * r.v[8] + l.v[6] * r.v[9] + l.v[10] * r.v[10] + l.v[14] * r.v[11],
            l.v[3] * r.v[8] + l.v[7] * r.v[9] + l.v[11] * r.v[10] + l.v[15] * r.v[11],

            l.v[0] * r.v[12] + l.v[4] * r.v[13] + l.v[8] * r.v[14] + l.v[12] * r.v[15],
            l.v[1] * r.v[12] + l.v[5] * r.v[13] + l.v[9] * r.v[14] + l.v[13] * r.v[15],
            l.v[2] * r.v[12] + l.v[6] * r.v[13] + l.v[10] * r.v[14] + l.v[14] * r.v[15],
            l.v[3] * r.v[12] + l.v[7] * r.v[13] + l.v[11] * r.v[14] + l.v[15] * r.v[15]
    };
}

kr_inline_host_device mat4  kr_mscreen4(u32 width, u32 height) {
    return KR_INITIALIZER_CAST(mat4) {
        (kr_scalar)width * .5f, 0.f, 0.f, 0.f,
            0.f, (kr_scalar)height * .5f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            (kr_scalar)width * .5f, (kr_scalar)height * .5f, 0.f, 1.f
    };
}

kr_inline_host_device kr_mat4
kr_mlookat4(kr_vec3 eye, kr_vec3 center, kr_vec3 up)
{
    kr_vec3 forward = kr_vnormalize3(kr_vsub3(center, eye));
    kr_vec3 right = kr_vnormalize3(kr_vcross3(up, forward));
    up = kr_vcross3(forward, right);

    return KR_INITIALIZER_CAST(kr_mat4) {
        right.x, up.x, forward.x, 0.0f,
        right.y, up.y, forward.y, 0.0f,
        right.z, up.z, forward.z, 0.0f,
        -kr_vdot3(right, eye), -kr_vdot3(up, eye), -kr_vdot3(forward, eye), 1.0f
    };
}

kr_inline_host_device vec3  kr_v43transform3p(const mat43* l, const vec3* r) {
    return KR_INITIALIZER_CAST(vec3) {
        l->v[0] * r->x + l->v[3] * r->y + l->v[6] * r->z + l->v[9],
        l->v[1] * r->x + l->v[4] * r->y + l->v[7] * r->z + l->v[10],
        l->v[2] * r->x + l->v[5] * r->y + l->v[8] * r->z + l->v[11],
    };
}

kr_inline_host_device vec3  kr_vtransform3p(const mat4* l, const vec3* r) {
    return KR_INITIALIZER_CAST(vec3) {
        l->v[0] * r->x + l->v[4] * r->y + l->v[8] * r->z + l->v[12],
        l->v[1] * r->x + l->v[5] * r->y + l->v[9] * r->z + l->v[13],
        l->v[2] * r->x + l->v[6] * r->y + l->v[10] * r->z + l->v[14],
    };
}

kr_inline_host_device vec3 kr_mvmul43(mat4 l, vec3 r) {
    return KR_INITIALIZER_CAST(vec3) {
        l.v[0] * r.x + l.v[4] * r.y + l.v[8] * r.z + l.v[12],
            l.v[1] * r.x + l.v[5] * r.y + l.v[9] * r.z + l.v[13],
            l.v[2] * r.x + l.v[6] * r.y + l.v[10] * r.z + l.v[14],
    };
}

kr_inline_host_device vec3 kr_vtransform3(mat4 l, vec3 r) {
    return kr_mvmul43(l, r);
}

kr_inline_host_device vec4 kr_mvmul44(mat4 m, vec4 v) {
    return KR_INITIALIZER_CAST(vec4) {
        v.x* m.v[0] + v.y * m.v[4] + v.z * m.v[8] + v.w * m.v[12],
            v.x* m.v[1] + v.y * m.v[5] + v.z * m.v[9] + v.w * m.v[13],
            v.x* m.v[2] + v.y * m.v[6] + v.z * m.v[10] + v.w * m.v[14],
            v.x* m.v[3] + v.y * m.v[7] + v.z * m.v[11] + v.w * m.v[15]
    };
}

kr_inline_host_device vec4 kr_vtransform4(mat4 l, vec4 r) {
    return kr_mvmul44(l, r);
}

kr_inline_host_device vec3  kr_n43transform3p(const mat43* l, const vec3* r) {
    return KR_INITIALIZER_CAST(vec3) {
        l->v[0] * r->x + l->v[3] * r->y + l->v[6] * r->z,
        l->v[1] * r->x + l->v[4] * r->y + l->v[7] * r->z,
        l->v[2] * r->x + l->v[5] * r->y + l->v[8] * r->z,
    };
}

kr_inline_host_device vec3  kr_ntransform3p(const mat4* l, const vec3* r) {
    return KR_INITIALIZER_CAST(vec3) {
        l->v[0] * r->x + l->v[4] * r->y + l->v[8] * r->z,
        l->v[1] * r->x + l->v[5] * r->y + l->v[9] * r->z,
        l->v[2] * r->x + l->v[6] * r->y + l->v[10] * r->z,
    };
}

kr_inline_host_device vec3  kr_ntransform3(mat4 l, vec3 r) {
    return KR_INITIALIZER_CAST(vec3) {
        l.v[0] * r.x + l.v[4] * r.y + l.v[8] * r.z,
            l.v[1] * r.x + l.v[5] * r.y + l.v[9] * r.z,
            l.v[2] * r.x + l.v[6] * r.y + l.v[10] * r.z,
    };
}

kr_inline_host_device vec3  kr_voffset3(vec3 o, vec3 d, kr_scalar s) {
    return kr_vadd3(o, kr_vmul31(d, s));
}

kr_inline_host_device vec3  kr_vproject3(mat4 l, vec3 r) {
    vec4 v = { r.x, r.y, r.z, 1.0f };

    v = kr_vtransform4(l, v);

    return KR_INITIALIZER_CAST(vec3) {
        v.x / v.w,
            v.y / v.w,
            v.z / v.w
    };
}

kr_inline_host_device vec3  kr_vadd31(vec3 a, kr_scalar b) {
    return KR_INITIALIZER_CAST(vec3) { a.x + b, a.y + b, a.z + b };
}

kr_inline_host_device vec3  kr_vmul31(vec3 a, kr_scalar b) {
    return KR_INITIALIZER_CAST(vec3) { a.x* b, a.y* b, a.z* b };
}

kr_inline_host_device vec2  kr_vmul21(vec2 a, kr_scalar b) {
    return KR_INITIALIZER_CAST(vec2) { a.x * b, a.y * b };
}

kr_inline_host_device kr_scalar kr_copysign(kr_scalar a, kr_scalar b) {
    return a * (((kr_scalar)0 < b) - (b < (kr_scalar)0));
}
kr_inline_host_device kr_scalar kr_sign(kr_scalar a) {
    return  (kr_scalar) (((kr_scalar)0.0 < a) - (a < (kr_scalar)0.0));
}
kr_inline_host_device kr_vec3 kr_vsign3(kr_vec3 v) {
    return KR_INITIALIZER_CAST(kr_vec3) {
        kr_sign(v.x),
        kr_sign(v.y),
        kr_sign(v.z)
    };
}
kr_inline_host_device kr_i32 kr_signi(kr_i32 a) {
    return (0 < a) - (a < 0);
}

kr_inline_host_device kr_scalar kr_lerp(kr_scalar x, kr_scalar y, kr_scalar a) {
    return ((kr_scalar)1 - a) * x + a * y;
}

kr_inline_host_device kr_vec4 kr_vlerp4(kr_vec4 x, kr_vec4 y, kr_scalar a) {
    return KR_INITIALIZER_CAST(kr_vec4) {
        ((kr_scalar)1 - a) * x.x + a * y.x,
        ((kr_scalar)1 - a) * x.y + a * y.y,
        ((kr_scalar)1 - a) * x.z + a * y.z,
        ((kr_scalar)1 - a) * x.w + a * y.w
    };
}

kr_inline_host_device kr_vec3 kr_vlerp3(kr_vec3 x, kr_vec3 y, kr_scalar a) {
    return KR_INITIALIZER_CAST(kr_vec3) {
        ((kr_scalar)1 - a) * x.x + a * y.x,
        ((kr_scalar)1 - a) * x.y + a * y.y,
        ((kr_scalar)1 - a) * x.z + a * y.z
    };
}

kr_inline_host_device kr_scalar kr_radians(kr_scalar dgrs) { return dgrs * (kr_scalar)0.01745329251994329576923690768489; }
kr_inline_host_device kr_scalar kr_degrees(kr_scalar rads) { return rads * (kr_scalar)57.295779513082320876798154814105; }

kr_inline_host_device kr_scalar kr_min(kr_scalar a, kr_scalar b) { return (a < b) ? a : b; }
kr_inline_host_device kr_scalar kr_max(kr_scalar a, kr_scalar b) { return (a < b) ? b : a; }
kr_inline_host_device kr_f64 kr_minf64(kr_f64 a, kr_f64 b) { return (a < b) ? a : b; }
kr_inline_host_device kr_f64 kr_maxf64(kr_f64 a, kr_f64 b) { return (a < b) ? b : a; }
kr_inline_host_device kr_i32 kr_mini(kr_i32 a, kr_i32 b) { return (a < b) ? a : b; }
kr_inline_host_device kr_i32 kr_maxi(kr_i32 a, kr_i32 b) { return (a < b) ? b : a; }
kr_inline_host_device kr_u32 kr_minu(kr_u32 a, kr_u32 b) { return (a < b) ? a : b; }
kr_inline_host_device kr_u32 kr_maxu(kr_u32 a, kr_u32 b) { return (a < b) ? b : a; }
kr_inline_host_device kr_scalar kr_abs(kr_scalar a) { return (a < 0.0) ? -a : a; }
kr_inline_host_device kr_b32    kr_isinf(kr_scalar a) { return isinf(a); };
kr_inline_host_device kr_scalar kr_cos(kr_scalar a) { return (kr_scalar)cos(a); }
kr_inline_host_device kr_scalar kr_sin(kr_scalar a) { return (kr_scalar)sin(a); }
kr_inline_host_device kr_scalar kr_sqrt(kr_scalar a) { return (kr_scalar)sqrt(a); }
kr_inline_host_device kr_scalar kr_min3(kr_vec3 a) { return kr_min(kr_min(a.x, a.y), a.z); }
kr_inline_host_device kr_scalar kr_max3(kr_vec3 a) { return kr_max(kr_max(a.x, a.y), a.z); }

kr_inline_host_device kr_bvec3 kr_vepsilon_equal3(kr_vec3 a, kr_vec3 b, kr_scalar eps) {
    return KR_INITIALIZER_CAST(kr_bvec3) {
        kr_abs(a.x - b.x) < eps,
        kr_abs(a.y - b.y) < eps,
        kr_abs(a.z - b.z) < eps
    };
}

kr_inline_host_device kr_b32 kr_vany3(kr_bvec3 a) {
    return (a.x || a.y || a.z);
}

kr_inline_host_device vec3 kr_vmin3(vec3 a, vec3 b) { return KR_INITIALIZER_CAST(vec3) { kr_min(a.x, b.x), kr_min(a.y, b.y), kr_min(a.z, b.z) }; }
kr_inline_host_device vec3 kr_vmax3(vec3 a, vec3 b) { return KR_INITIALIZER_CAST(vec3) { kr_max(a.x, b.x), kr_max(a.y, b.y), kr_max(a.z, b.z) }; }

kr_inline_host_device void kr_onb_create(kr_vec3 vector, kr_onb* onb) {
    onb->normal = vector;

    kr_scalar sign = (vector.z >= 0.0) ? 1.0f : -1.0f;
    kr_scalar a = -1.0f / (sign + vector.z);
    kr_scalar b = vector.x * vector.y * a;
    onb->tangent = KR_INITIALIZER_CAST(kr_vec3) { 1.f + sign * vector.x * vector.x * a, sign * b, -sign * vector.x };
    onb->bitangent = KR_INITIALIZER_CAST(kr_vec3) { b, sign + vector.y * vector.y * a, -vector.y };
}

kr_inline_host_device kr_vec3 kr_onb_vto3(kr_onb* onb, kr_vec3 v) {
    return KR_INITIALIZER_CAST(vec3) {
            kr_vdot3(onb->tangent, v),
            kr_vdot3(onb->bitangent, v),
            kr_vdot3(onb->normal, v) };
}

kr_inline_host_device kr_vec3 kr_onb_vfrom3(kr_onb* onb, kr_vec3 v) {
    return 
      kr_vadd3(
        kr_vadd3(
          kr_vmul31(onb->tangent, v.x), 
          kr_vmul31(onb->bitangent, v.y)
        ), 
        kr_vmul31(onb->normal, v.z)
      );
}

kr_inline_host_device kr_scalar kr_quantize(kr_scalar a) {
    return (kr_scalar)((int)(a * 127.0f + (a > 0 ? 0.5f : -0.5f))) / 127.0f;
}

kr_inline_host_device kr_vec3 kr_vquantize3(kr_vec3 v) {
    return KR_INITIALIZER_CAST(vec3) {
        kr_quantize(v.x),
        kr_quantize(v.y),
        kr_quantize(v.z)
    };
}

kr_inline_host_device kr_u32    kr_ctz32(kr_u32 value) {
    kr_u32 trailing_zero = 0;
#if KR_LINUX 
    return __builtin_ctz(value);
#elif KR_WIN32
    if (_BitScanForward((unsigned long*)&trailing_zero, value))
    {
        return trailing_zero;
    }
    else
    {
        // This is undefined, I better choose 32 than 0
        return 32;
    }
#endif

    return trailing_zero;
}

kr_inline_host_device kr_u64    kr_ctz64(kr_u64 value) {
    kr_u64 trailing_zero = 0;
#if KR_LINUX
    return __builtin_ctz(value);
#elif KR_WIN32
    if (_BitScanForward64((unsigned long*)&trailing_zero, value))
    {
        return trailing_zero;
    }
    else
    {
        // This is undefined, I better choose 32 than 0
        return 32;
    }
#endif

    return trailing_zero;
}

kr_inline_host_device kr_u32    kr_clz32(kr_u32 value) {
    kr_u32 leading_zero = 0;
#if KR_LINUX 
    return __builtin_clz(value);
#elif KR_WIN32
    if (_BitScanReverse((unsigned long*)&leading_zero, value))
    {
        return 31 - leading_zero;
    }
    else
    {
        // Same remarks as above
        return 32;
    }
#endif /* KR_WIN32 */

    return leading_zero;
}


kr_inline_host_device kr_u64    kr_clz64(kr_u64 value) {
    kr_u64 leading_zero = 0;

#if KR_LINUX
    return __builtin_clzll(value);
#elif KR_WIN32
    if (_BitScanReverse64((unsigned long*)&leading_zero, value))
    {
        return 63 - leading_zero;
    }
    else
    {
        // Same remarks as above
        return 64;
    }
#endif /* KR_WIN32 */

    return leading_zero;
}

kr_inline_host_device vec3 kr_vinv3(vec3 a) { 
    return KR_INITIALIZER_CAST(vec3) { (kr_scalar)1 / a.x, (kr_scalar)1 / a.y, (kr_scalar)1 / a.z };
}

kr_inline_host_device vec3 kr_vsafeinv3(vec3 a) {
    const kr_scalar ooeps = (kr_scalar)1e-8;
    
    return KR_INITIALIZER_CAST(vec3) { 
        (kr_scalar)(1 / (kr_abs(a.x) > ooeps ? a.x : copysign(ooeps, a.x))),
        (kr_scalar)(1 / (kr_abs(a.y) > ooeps ? a.y : copysign(ooeps, a.y))),
        (kr_scalar)(1 / (kr_abs(a.z) > ooeps ? a.z : copysign(ooeps, a.z)))
    };
}

kr_inline_host_device vec3 kr_aabb_center3(aabb3 box) {
    return kr_vmul31(kr_vadd3(box.min, box.max), (kr_scalar)0.5);
}

kr_inline_host_device kr_scalar kr_aabb_radius3(aabb3 box) {
    return kr_max3(kr_aabb_extents3(box)) * (kr_scalar)0.5;
}

kr_inline_host_device kr_aabb3 kr_aabb_empty3() {
    return KR_INITIALIZER_CAST(kr_aabb3) { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
}

kr_inline_host_device vec3
kr_obb_extents3(kr_obb3 obb) {
    return KR_INITIALIZER_CAST(vec3) { 2.0f * obb.ext.x, 2.0f * obb.ext.y, 2.0f * obb.ext.z };
}

kr_inline_host_device kr_scalar 
kr_obb_surface_area3(kr_obb3 obb) {
    kr_vec3 d = kr_obb_extents3(obb);
    return (kr_scalar)(2) * (d.x * d.y + d.x * d.z + d.y * d.z);
}

kr_inline_host_device kr_scalar 
kr_aabb_surface_area3(aabb3 box) {
	kr_vec3 d = kr_aabb_extents3(box);
	return (kr_scalar)(2) * (d.x * d.y + d.x * d.z + d.y * d.z);
}

kr_inline_host_device kr_scalar 
kr_aabb_volume3(aabb3 box) {
    kr_vec3 d = kr_aabb_extents3(box);
    return d.x * d.y * d.z;
}

kr_inline_host_device vec3 kr_aabb_extents3(aabb3 box) {
    return KR_INITIALIZER_CAST(vec3) { box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z };
}

kr_inline_host_device aabb3 kr_aabb_shrink3(aabb3 box, kr_scalar v) {
    return KR_INITIALIZER_CAST(aabb3) { box.min.x + v, box.min.y + v, box.min.z + v, box.max.x - v, box.max.y - v, box.max.z - v };
}

kr_inline_host_device aabb3 kr_aabb_blow3(aabb3 box, kr_scalar v) {
    return KR_INITIALIZER_CAST(aabb3) { box.min.x - v, box.min.y - v, box.min.z - v, box.max.x + v, box.max.y + v, box.max.z + v };
}

kr_inline_host_device aabb3 kr_aabb_create3(vec3 vmin, vec3 vmax) {
    return KR_INITIALIZER_CAST(aabb3) { vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z };
}

kr_inline_host_device void kr_aabb_split3(kr_aabb3 box, kr_aabb3* children) {
    kr_aabb3 cell         = kr_aabb_create3(box.min, kr_aabb_center3(box));
    vec3     cell_extents = kr_aabb_extents3(cell);

    children[0] = cell;
    children[1] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {cell_extents.x, 0, 0});
    children[2] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {0, cell_extents.y, 0});
    children[3] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {0, 0, cell_extents.z});
    children[4] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {cell_extents.x, cell_extents.y, 0});
    children[5] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {cell_extents.x, 0, cell_extents.z});
    children[6] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {0, cell_extents.y, cell_extents.z});
    children[7] = kr_aabb_offset3(cell, KR_INITIALIZER_CAST(kr_vec3) {cell_extents.x, cell_extents.y, cell_extents.z});
}

kr_inline_host_device void kr_aabb_corners3(kr_aabb3 box, kr_vec3* points) {

    points[0] = box.min;
    points[1] = KR_INITIALIZER_CAST(kr_vec3) { box.max.x, box.min.y, box.min.z };
    points[2] = KR_INITIALIZER_CAST(kr_vec3) { box.min.x, box.max.y, box.min.z };
    points[3] = KR_INITIALIZER_CAST(kr_vec3) { box.min.x, box.min.y, box.max.z };
    points[4] = KR_INITIALIZER_CAST(kr_vec3) { box.max.x, box.min.y, box.max.z };
    points[5] = KR_INITIALIZER_CAST(kr_vec3) { box.max.x, box.max.y, box.min.z };
    points[6] = KR_INITIALIZER_CAST(kr_vec3) { box.min.x, box.max.y, box.max.z };
    points[7] = box.max;
}

kr_inline_host_device aabb3 kr_aabb_offset3(aabb3 box, vec3 a) {
    return KR_INITIALIZER_CAST(aabb3) { box.min.x + a.x, box.min.y + a.y, box.min.z + a.z, box.max.x + a.x, box.max.y + a.y, box.max.z + a.z };
}

kr_inline_host_device aabb3 kr_aabb_expand3(aabb3 box, vec3 a) {
    vec3 new_min = kr_vmin3(box.min, a);
    vec3 new_max = kr_vmax3(box.max, a);
    return KR_INITIALIZER_CAST(aabb3) { new_min.x, new_min.y, new_min.z, new_max.x, new_max.y, new_max.z };
}

kr_inline_host_device aabb3 kr_aabb_transform4(mat4 m, aabb3 box) {
    vec3 new_min = kr_vtransform3(m, box.min);
    vec3 new_max = kr_vtransform3(m, box.max);
    return KR_INITIALIZER_CAST(aabb3) { new_min.x, new_min.y, new_min.z, new_max.x, new_max.y, new_max.z };
}

kr_inline_host_device aabb3 kr_aabb_expand(aabb3 box, aabb3 a) {
    return kr_aabb_expand3(kr_aabb_expand3(box, a.min), a.max);
}

kr_inline_host_device vec3 kr_ray_triangle_intersect(vec3 direction, vec3 origin, vec3 v1, vec3 v2, vec3 v3, kr_scalar t_max)
{
    vec3 e1 = kr_vsub3(v2, v1);
    vec3 e2 = kr_vsub3(v3, v1);
    vec3 s1 = kr_vcross3(direction, e2);

    kr_scalar invd = 1.0f / kr_vdot3(s1, e1);

    vec3 d = kr_vsub3(origin, v1);
    kr_scalar b1 = kr_vdot3(d, s1) * invd;
    vec3 s2 = kr_vcross3(d, e1);
    kr_scalar b2 = kr_vdot3(direction, s2) * invd;
    kr_scalar temp = kr_vdot3(e2, s2) * invd;
    if (b1 < 0.0 || b1 > 1.0 || b2 < 0.0 || b1 + b2 > 1.0 || temp < 0.0 || temp > t_max)
    {
        return KR_INITIALIZER_CAST(vec3) { 0, 0, t_max };
    }
    else
    {
        return KR_INITIALIZER_CAST(vec3) { b1, b2, temp };
    }
}

kr_inline_host_device void kr_swapf(kr_scalar* a, kr_scalar* b) {
    kr_scalar tmp = *a;
    *a = *b;
    *b = tmp;
}

kr_inline_host_device kr_vec2 kr_ray_unit_aabb_intersect_n(kr_vec3 rd, kr_vec3 ro, kr_scalar t_max, kr_vec3* normal) {
    kr_scalar tmin_x = ((kr_scalar)-0.5 - ro.x) / rd.x;
    kr_scalar tmin_y = ((kr_scalar)-0.5 - ro.y) / rd.y;
    kr_scalar tmin_z = ((kr_scalar)-0.5 - ro.z) / rd.z;

    kr_scalar tmax_x = ((kr_scalar)0.5 - ro.x) / rd.x;
    kr_scalar tmax_y = ((kr_scalar)0.5 - ro.y) / rd.y;
    kr_scalar tmax_z = ((kr_scalar)0.5 - ro.z) / rd.z;

    kr_scalar sc_x = kr_min(tmin_x, tmax_x);
    kr_scalar sc_y = kr_min(tmin_y, tmax_y);
    kr_scalar sc_z = kr_min(tmin_z, tmax_z);

    kr_scalar sf_x = kr_max(tmin_x, tmax_x);
    kr_scalar sf_y = kr_max(tmin_y, tmax_y);
    kr_scalar sf_z = kr_max(tmin_z, tmax_z);

    kr_scalar t0 = kr_max(kr_max(sc_x, sc_y), sc_z);
    kr_scalar t1 = kr_min(kr_min(sf_x, sf_y), sf_z);
    if (!(t0 <= t1 && t1 > 0.0)) return KR_INITIALIZER_CAST(vec2) { t_max, -t_max };

    cvec3 sc_yzx = { sc_y, sc_z, sc_x };
    cvec3 sc_zxy = { sc_z, sc_x, sc_y };

    *normal = kr_vmul3(
        kr_vnegate3(kr_vsign3(rd)),
        kr_vmul3(
            kr_vstep3(sc_yzx, KR_INITIALIZER_CAST(vec3) { sc_x, sc_y, sc_z }),
            kr_vstep3(sc_zxy, KR_INITIALIZER_CAST(vec3) { sc_x, sc_y, sc_z })
        ));

    return KR_INITIALIZER_CAST(vec2) { t0, t1 };
}

kr_inline_host_device kr_vec2 kr_ray_unit_aabb_intersect(kr_vec3 rd, kr_vec3 ro, kr_scalar t_max) {
#if 0
    kr_scalar tmin_x = (-0.5 - ro.x) / rd.x;
    kr_scalar tmin_y = (-0.5 - ro.y) / rd.y;
    kr_scalar tmin_z = (-0.5 - ro.z) / rd.z;

    kr_scalar tmax_x = ( 0.5 - ro.x) / rd.x;
    kr_scalar tmax_y = ( 0.5 - ro.y) / rd.y;
    kr_scalar tmax_z = ( 0.5 - ro.z) / rd.z;

    kr_scalar sc_x = kr_min(tmin_x, tmax_x);
    kr_scalar sc_y = kr_min(tmin_y, tmax_y);
    kr_scalar sc_z = kr_min(tmin_z, tmax_z);

    kr_scalar sf_x = kr_max(tmin_x, tmax_x);
    kr_scalar sf_y = kr_max(tmin_y, tmax_y);
    kr_scalar sf_z = kr_max(tmin_z, tmax_z);

    kr_scalar t0 = kr_max(kr_max(sc_x, sc_y), sc_z);
    kr_scalar t1 = kr_min(kr_min(sf_x, sf_y), sf_z);
    if (!(t0 <= t1 && t1 > 0.0)) return KR_INITIALIZER_CAST(vec2) { t_max, -t_max };

    return KR_INITIALIZER_CAST(vec2) { t0, t1 };
#else
    // Basic ray/AABB intersection (see Section 32.3)
    vec3 tmin = kr_vdiv3(kr_vsub3(KR_INITIALIZER_CAST(vec3) {-0.5f, -0.5f, -0.5f }, ro), rd);
    vec3 tmax = kr_vdiv3(kr_vsub3(KR_INITIALIZER_CAST(vec3) { 0.5f,  0.5f,  0.5f }, ro), rd);

    vec3 sc = kr_vmin3(tmin, tmax);
    vec3 sf = kr_vmax3(tmin, tmax);
    kr_scalar t0 = kr_max(kr_max(sc.x, sc.y), sc.z);
    kr_scalar t1 = kr_min(kr_min(sf.x, sf.y), sf.z);
    if (!(t0 <= t1 && t1 > 0.0)) return KR_INITIALIZER_CAST(vec2) { 1, -1 };

    return KR_INITIALIZER_CAST(vec2) { t0, t1 };
#endif
}


kr_inline_host_device kr_vec2 kr_ray_aabb_intersect_n(kr_vec3 rd, kr_vec3 ro, kr_aabb3 bbox, kr_scalar t_max, kr_vec3* normal)
{
    // Basic ray/AABB intersection (see Section 32.3)
    vec3 tmin = kr_vdiv3(kr_vsub3(bbox.min, ro), rd);
    vec3 tmax = kr_vdiv3(kr_vsub3(bbox.max, ro), rd);

    vec3 sc = kr_vmin3(tmin, tmax);
    vec3 sf = kr_vmax3(tmin, tmax);
    float t0 = kr_max(kr_max(sc.x, sc.y), sc.z);
    float t1 = kr_min(kr_min(sf.x, sf.y), sf.z);
    if (!(t0 <= t1 && t1 > 0.0)) return KR_INITIALIZER_CAST(vec2) { t_max, -t_max };
    
    // Computing additional intersection data (see Section 32.5)
    // Normals

    cvec3 sc_yzx = { sc.y, sc.z, sc.x };
    cvec3 sc_zxy = { sc.z, sc.x, sc.y };

    *normal = kr_vmul3(
        kr_vnegate3(kr_vsign3(rd)),
        kr_vmul3(
            kr_vstep3(sc_yzx, sc),
            kr_vstep3(sc_zxy, sc)
        ));

    return KR_INITIALIZER_CAST(vec2) { t0, t1 };
}

kr_inline_host_device kr_vec2 kr_ray_aabb_intersect(kr_vec3 rd, kr_vec3 ro, kr_aabb3 bbox, kr_scalar t_max)
{
    // Basic ray/AABB intersection (see Section 32.3)
    vec3 tmin = kr_vdiv3(kr_vsub3(bbox.min, ro), rd);
    vec3 tmax = kr_vdiv3(kr_vsub3(bbox.max, ro), rd);

    vec3 sc = kr_vmin3(tmin, tmax);
    vec3 sf = kr_vmax3(tmin, tmax);
    float t0 = kr_max(kr_max(sc.x, sc.y), sc.z);
    float t1 = kr_min(kr_min(sf.x, sf.y), sf.z);
    if (!(t0 <= t1 && t1 > 0.0)) return KR_INITIALIZER_CAST(vec2) { 1, -1 };
   
    return KR_INITIALIZER_CAST(vec2) { t0, t1 };
}

#endif /* KR_VECMATH_IMPL */

#endif /* KR_VECMATH_H */
