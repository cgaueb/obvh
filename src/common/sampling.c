#define KR_SAMPLING_IMPL
#include "sampling.h"

#include "util.h"

kr_error
kr_piecewise_constant_1D_create(kr_piecewise_constant_1D* dist, kr_scalar* data, kr_size count) {
    *dist = (kr_piecewise_constant_1D){ 0 };

    dist->count = count;
    dist->func = (kr_scalar*)kr_allocate(sizeof(*dist->func) * count);
    dist->cdf  = (kr_scalar*)kr_allocate(sizeof(*dist->cdf) * (count + 1));

    kr_memcpy(dist->func, data, sizeof(*data) * count);

    for (kr_size i = 0; i < count; ++i) {
        dist->func[i] = kr_abs(dist->func[i]);
    }

    dist->cdf[0] = 0;
    for (kr_size i = 0; i < count + 1; ++i) {
        dist->func[i] = kr_abs(dist->func[i]);
        dist->cdf[i] = dist->cdf[i - 1] + dist->func[i - 1] / count;
    }

    // Transform step function integral into CDF
    dist->funcInt = dist->cdf[count];
    if (dist->funcInt == 0)
        for (kr_size i = 1; i < count + 1; ++i)
            dist->cdf[i] = (kr_scalar)i / (kr_scalar)count;
    else
        for (kr_size i = 1; i < count + 1; ++i)
            dist->cdf[i] /= dist->funcInt;

    return kr_success;
}

#if 0
kr_vec2 
kr_concentric_disk_sample(kr_vec2 u) {
    // Map uniform random numbers to $[-1,1]^2$
    kr_vec2 uOffset = kr_vsub2(kr_vmul21(u, 2.f), KR_INITIALIZER_CAST(vec2) { 1, 1 });

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return KR_INITIALIZER_CAST(vec2) { 0, 0 };

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

kr_vec3 
kr_hemisphere_consine_sample(kr_vec2 u) {
    kr_vec2 d = kr_concentric_disk_sample(u);
    kr_scalar z = kr_sqrt(kr_max((kr_scalar)0, 1 - d.x * d.x - d.y * d.y));
    return KR_INITIALIZER_CAST(vec3) { d.x, d.y, z };
}

kr_vec3 
kr_sphere_uniform_sample(kr_vec2 u) {
    kr_scalar z = 1 - 2 * u.v[0];
    kr_scalar r = kr_sqrt(1 - z * z);
    kr_scalar phi = 2 * KR_PI * u.v[1];

    return KR_INITIALIZER_CAST(kr_vec3) { r * kr_cos(phi), r * kr_sin(phi), z };
}

kr_vec3
kr_hemisphere_uniform_sample(kr_vec2 u) {
    kr_scalar z = u.v[0];
    kr_scalar r = kr_sqrt(1 - (z * z));
    kr_scalar phi = 2 * KR_PI * u.v[1];
    return KR_INITIALIZER_CAST(kr_vec3) { r * kr_cos(phi), r * kr_sin(phi), z };
}

kr_scalar
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

kr_error
kr_piecewise_constant_1D_create(kr_piecewise_constant_1D* dist, kr_scalar* data, kr_size count) {
    *dist = (kr_piecewise_constant_1D){ 0 };

    dist->count = count;
    dist->func = (kr_scalar*)kr_allocate(sizeof(*dist->func) * count);
    dist->cdf  = (kr_scalar*)kr_allocate(sizeof(*dist->cdf) * (count + 1));

    kr_memcpy(dist->func, data, sizeof(*data) * count);

    for (kr_size i = 0; i < count; ++i) {
        dist->func[i] = kr_abs(dist->func[i]);
    }

    dist->cdf[0] = 0;
    for (kr_size i = 0; i < count + 1; ++i) {
        dist->func[i] = kr_abs(dist->func[i]);
        dist->cdf[i] = dist->cdf[i - 1] + dist->func[i - 1] / count;
    }

    // Transform step function integral into CDF
    dist->funcInt = dist->cdf[count];
    if (dist->funcInt == 0)
        for (kr_size i = 1; i < count + 1; ++i)
            dist->cdf[i] = (kr_scalar)i / (kr_scalar)count;
    else
        for (kr_size i = 1; i < count + 1; ++i)
            dist->cdf[i] /= dist->funcInt;

    return kr_success;
}

kr_vec2 
kr_triangle_uniform_sample(kr_vec2 u) {
    kr_scalar su0 = kr_sqrt(u.v[0]);
    return KR_INITIALIZER_CAST(kr_vec2) { 1 - su0, u.v[1] * su0 };
}
#endif