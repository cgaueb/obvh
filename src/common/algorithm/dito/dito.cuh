#ifndef _KORANGAR_DITO_CUDA_H_
#define _KORANGAR_DITO_CUDA_H_

#include "common/vecmath.h"

#include "common/algorithm/dito/dito.h"

#ifdef __cplusplus
extern "C" {
#endif

kr_obb3
kr_cuda_points_obb(const kr_vec3* points, kr_size count, kr_scalar* elapsed_time);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_DITO_CUDA_H_ */
