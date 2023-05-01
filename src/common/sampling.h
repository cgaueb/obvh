#ifndef _KORANGAR_SAMPLING_H_
#define _KORANGAR_SAMPLING_H_

#include "korangar.h"

#ifdef __cplusplus
extern "C" {
#endif

    kr_vec2
        kr_concentric_disk_sample(kr_vec2 u);
    kr_vec3
        kr_hemisphere_consine_sample(kr_vec2 u);
    kr_vec3
        kr_sphere_uniform_sample(kr_vec2 u);
    kr_vec3
        kr_hemisphere_uniform_sample(kr_vec2 u);
    kr_vec2
        kr_triangle_uniform_sample(kr_vec2 u);

    typedef struct {
        kr_size    count;
        kr_scalar* func;
        kr_scalar* cdf;
        kr_scalar  funcInt;
    } kr_piecewise_constant_1D;

    kr_error
        kr_piecewise_constant_1D_create(kr_piecewise_constant_1D* dist, kr_scalar* data, kr_size count);
    kr_scalar
        kr_piecewise_constant_1D_sample(kr_piecewise_constant_1D* dist, kr_scalar u);

#ifdef __cplusplus
}
#endif

#if defined(KR_SAMPLING_IMPL)

kr_inline_device kr_vec3
kr_hemisphere_consine_sample(kr_vec2 u) {
    kr_vec2 d = kr_concentric_disk_sample(u);
    kr_scalar z = kr_sqrt(kr_max((kr_scalar)0, 1 - d.x * d.x - d.y * d.y));
    return KR_INITIALIZER_CAST(vec3) { d.x, d.y, z };
}

kr_inline_device kr_vec2
kr_concentric_disk_sample(kr_vec2 u) {
    // Map uniform random numbers to $[-1,1]^2$
    kr_vec2 uOffset = kr_vsub2(kr_vmul21(u, 2.f), KR_INITIALIZER_CAST(vec2) { 1.0, 1.0 });

    // Handle degeneracy at the origin
    if (uOffset.x == 0.0 && uOffset.y == 0.0) return KR_INITIALIZER_CAST(vec2) { 0.0, 0.0 };

    // Apply concentric mapping to point
    kr_scalar theta, r;
    if (kr_abs(uOffset.x) > kr_abs(uOffset.y)) {
        r = uOffset.x;
        theta = KR_PI_OVER_4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = KR_PI_OVER_2 - KR_PI_OVER_4 * (uOffset.x / uOffset.y);
    }
    return kr_vmul21(KR_INITIALIZER_CAST(vec2) { kr_cos(theta), kr_sin(theta) }, r);
}


kr_inline_device kr_vec3 
kr_sphere_uniform_sample(kr_vec2 u) {
    kr_scalar z = 1 - 2 * u.v[0];
    kr_scalar r = kr_sqrt(1 - z * z);
    kr_scalar phi = 2 * KR_PI * u.v[1];

    return KR_INITIALIZER_CAST(kr_vec3) { r * kr_cos(phi), r * kr_sin(phi), z };
}

kr_inline_device kr_vec3
kr_hemisphere_uniform_sample(kr_vec2 u) {
    kr_scalar z = u.v[0];
    kr_scalar r = kr_sqrt(1 - (z * z));
    kr_scalar phi = 2 * KR_PI * u.v[1];
    return KR_INITIALIZER_CAST(kr_vec3) { r * kr_cos(phi), r * kr_sin(phi), z };
}

kr_inline_device kr_vec2 
kr_triangle_uniform_sample(kr_vec2 u) {
    kr_scalar su0 = kr_sqrt(u.v[0]);
    return KR_INITIALIZER_CAST(kr_vec2) { 1 - su0, u.v[1] * su0 };
}

kr_inline_device kr_scalar
kr_piecewise_constant_1D_sample(kr_piecewise_constant_1D* dist, kr_scalar u) {
    kr_isize size = dist->count - 2, first = 1;
    while (size > 0) {
        // Evaluate predicate at midpoint and update _first_ and _size_
        kr_size half = (kr_size)size >> 1, middle = first + half;
        b32 predResult = dist->cdf[middle] <= u;
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }
    
    kr_size off = (kr_size)kr_clampi((kr_isize)first - 1, 0, dist->count - 2);

    // Compute offset along CDF segment
    kr_scalar du = u - dist->cdf[off];
    if (dist->cdf[off + 1] - dist->cdf[off] > 0)
        du /= dist->cdf[off + 1] - dist->cdf[off];
    
    return kr_lerp(0.0, 1.0, (off + du) / dist->count);
}


#endif /* KR_SAMPLING_IMPL */

#endif /* _KORANGAR_SAMPLING_H_ */
