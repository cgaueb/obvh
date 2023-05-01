#ifndef _KORANGAR_TEXTURE_H_
#define _KORANGAR_TEXTURE_H_

#include "korangar.h"
#include "vecmath.h"

#ifdef __cplusplus
extern "C" {
#endif

	kr_vec4
		kr_texture_2d_sample_nearest(kr_texture* texture, kr_vec2 uv);
	kr_vec4
		kr_texture_2d_sample(kr_texture* texture, kr_vec2 uv);
	kr_vec3
		kr_texture_2d_sample3(kr_texture* texture, kr_sampler* sampler, kr_vec2 uv);
	kr_vec4
		kr_texture_2d_sample4(kr_texture* texture, kr_vec2 uv);
	u32
		kr_texture_memory_size(kr_texture_type type, kr_uvec3 dims);
	u8*
		kr_texture_allocate(kr_texture_type type, kr_uvec3 dims);
	kr_error
		kr_texture_destroy(kr_texture* texture);
	kr_error
		kr_texture_create_from_file(kr_texture* texture, const char* filename);
	kr_error
		kr_texture_save_png(kr_texture* texture, const char* filename, b32 flip);
	kr_error
		kr_texture_save_tiff(kr_texture* texture, const char* filename, b32 flip);

#ifdef __cplusplus
}
#endif

#endif /* _KORANGAR_SAMPLING_H_ */
