#include "texture.h"
#include "util.h"

#ifdef KR_HAS_FREEIMAGE
#include <FreeImage.h>
#endif

kr_internal kr_vec4
kr_texture_2d_rgba8u_sample_nearest(kr_texture* texture, kr_vec2 uv) {
    kr_u8vec4* data = (kr_u8vec4*)texture->data;

    uvec2 udims = { texture->dims.x, texture->dims.y };
    kr_scalar vx = uv.x * (kr_scalar)udims.x;
    kr_scalar vy = uv.y * (kr_scalar)udims.y;
    uvec2 coords = { (u32)vx, (u32)vy };
    coords = kr_uvclamp2(coords, (uvec2) { 0, 0 }, (uvec2) { udims.x - 1, udims.y - 1 });
    u32 index = coords.y * udims.x + coords.x;

    kr_u8vec4 pixel = { data[index].r, data[index].g, data[index].b, data[index].a };

    return kr_u8vnorm4(pixel);
}

kr_internal kr_vec4
kr_texture_2d_rgb8u_sample_nearest(kr_texture* texture, kr_vec2 uv) {
  kr_u8vec3* data = (kr_u8vec3*)texture->data;

  uvec2 udims = {texture->dims.x, texture->dims.y};
  kr_scalar vx = uv.x * (kr_scalar) udims.x;
  kr_scalar vy = uv.y * (kr_scalar) udims.y;
  uvec2 coords = { (u32)vx, (u32)vy};
  coords = kr_uvclamp2(coords, (uvec2) {0, 0}, (uvec2) {udims.x - 1, udims.y - 1});
  u32 index = coords.y * udims.x + coords.x;
  
  kr_u8vec4 pixel = {data[index].r, data[index].g, data[index].b, 0};

  return kr_u8vnorm4(pixel);
}


kr_internal kr_vec4
kr_texture_2d_rgba8u_sample(kr_texture* texture, kr_vec2 uv) {
    kr_u8vec4* data = (kr_u8vec4*)texture->data;

    uvec2 udims = { texture->dims.x, texture->dims.y };
    kr_scalar vx = uv.x * (kr_scalar)udims.x;
    kr_scalar vy = uv.y * (kr_scalar)udims.y;
    u32 x = (u32)vx;
    u32 y = (u32)vy;
    vec2 coords = { vx, vy };
    vec2 coords_fract = kr_vfract2(coords);

    uvec2 coords00 = { x + 0, y + 0 };
    uvec2 coords01 = { x + 0, y + 1 };
    uvec2 coords10 = { x + 1, y + 0 };
    uvec2 coords11 = { x + 1, y + 1 };
    coords00 = kr_uvclamp2(coords00, (uvec2) { 0, 0 }, (uvec2) { udims.x - 1, udims.y - 1 });
    coords01 = kr_uvclamp2(coords01, (uvec2) { 0, 0 }, (uvec2) { udims.x - 1, udims.y - 1 });
    coords10 = kr_uvclamp2(coords10, (uvec2) { 0, 0 }, (uvec2) { udims.x - 1, udims.y - 1 });
    coords11 = kr_uvclamp2(coords11, (uvec2) { 0, 0 }, (uvec2) { udims.x - 1, udims.y - 1 });

    u32 index00 = coords00.y * udims.x + coords00.x;
    u32 index01 = coords01.y * udims.x + coords01.x;
    u32 index10 = coords10.y * udims.x + coords10.x;
    u32 index11 = coords11.y * udims.x + coords11.x;

    kr_u8vec4 pixel00 = { data[index00].r, data[index00].g, data[index00].b, data[index00].a };
    kr_u8vec4 pixel01 = { data[index01].r, data[index01].g, data[index01].b, data[index01].a };
    kr_u8vec4 pixel10 = { data[index10].r, data[index10].g, data[index10].b, data[index10].a };
    kr_u8vec4 pixel11 = { data[index11].r, data[index11].g, data[index11].b, data[index11].a };

    kr_u8vec4 pixel_tA = kr_u8vmix4(pixel00, pixel10, coords_fract.x);
    kr_u8vec4 pixel_tB = kr_u8vmix4(pixel01, pixel11, coords_fract.x);
    kr_u8vec4 pixel    = kr_u8vmix4(pixel_tA, pixel_tB, coords_fract.y);
    kr_u8vec4 out      = { pixel.x, pixel.y, pixel.z, pixel.w };

    return kr_u8vnorm4(out);
}

kr_internal kr_vec4
kr_texture_2d_rgb8u_sample(kr_texture* texture, kr_vec2 uv) {
  kr_u8vec3* data = (kr_u8vec3*)texture->data;

  uvec2 udims = {texture->dims.x, texture->dims.y};
  kr_scalar vx = uv.x * (kr_scalar) udims.x;
  kr_scalar vy = uv.y * (kr_scalar) udims.y;
  u32 x = (u32)vx;
  u32 y = (u32)vy;
  vec2 coords = {vx, vy};
  vec2 coords_fract = kr_vfract2(coords);

  uvec2 coords00 = {x + 0, y + 0};
  uvec2 coords01 = {x + 0, y + 1};
  uvec2 coords10 = {x + 1, y + 0};
  uvec2 coords11 = {x + 1, y + 1};
  coords00 = kr_uvclamp2(coords00, (uvec2) {0, 0}, (uvec2) {udims.x - 1, udims.y - 1});
  coords01 = kr_uvclamp2(coords01, (uvec2) {0, 0}, (uvec2) {udims.x - 1, udims.y - 1});
  coords10 = kr_uvclamp2(coords10, (uvec2) {0, 0}, (uvec2) {udims.x - 1, udims.y - 1});
  coords11 = kr_uvclamp2(coords11, (uvec2) {0, 0}, (uvec2) {udims.x - 1, udims.y - 1});

  u32 index00 = coords00.y * udims.x + coords00.x;
  u32 index01 = coords01.y * udims.x + coords01.x;
  u32 index10 = coords10.y * udims.x + coords10.x;
  u32 index11 = coords11.y * udims.x + coords11.x;

  kr_u8vec3 pixel00 = {data[index00].r, data[index00].g, data[index00].b};
  kr_u8vec3 pixel01 = {data[index01].r, data[index01].g, data[index01].b};
  kr_u8vec3 pixel10 = {data[index10].r, data[index10].g, data[index10].b};
  kr_u8vec3 pixel11 = {data[index11].r, data[index11].g, data[index11].b};

  kr_u8vec3 pixel_tA = kr_u8vmix3(pixel00, pixel10, coords_fract.x);
  kr_u8vec3 pixel_tB = kr_u8vmix3(pixel01, pixel11, coords_fract.x);
  kr_u8vec3 pixel = kr_u8vmix3(pixel_tA, pixel_tB, coords_fract.y);
  kr_u8vec4 out = { pixel.x, pixel.y, pixel.z, 0};

  return kr_u8vnorm4(out);
}

u32
kr_texture_memory_size(kr_texture_type type, kr_uvec3 dims) {
  kr_uvec3 safe_dims = kr_uvclamp3(dims, (kr_uvec3){1,1,1}, dims);
  switch(type) {
    case KR_TEXTURE_2D_RGB8U:
      return (safe_dims.x * safe_dims.y * safe_dims.z * 3);
      break;
    case KR_TEXTURE_2D_RGBA8U:
      return (safe_dims.x * safe_dims.y * safe_dims.z * 4);
      break;
    default:
      return 0;
      break;
  }
  return 0;
}

u8*
kr_texture_allocate(kr_texture_type type, kr_uvec3 dims) {
  kr_uvec3 safe_dims = kr_uvclamp3(dims, (kr_uvec3){1,1,1}, dims);
  switch(type) {
    case KR_TEXTURE_2D_RGB8U:
      return kr_aligned_allocate(safe_dims.x * safe_dims.y * safe_dims.z * 3, 16);
      break;
    case KR_TEXTURE_2D_RGBA8U:
      return kr_aligned_allocate(safe_dims.x * safe_dims.y * safe_dims.z * 4, 16);
      break;
    default:
      return kr_null;
      break;
  }
  return kr_null;
}

kr_error
kr_texture_destroy(kr_texture* texture) {
  kr_aligned_free((void**)&texture->data);

  return kr_success;
}

kr_vec4
kr_texture_2d_sample_nearest(kr_texture* texture, kr_vec2 uv) {
  switch(texture->type) {
    case KR_TEXTURE_2D_RGB8U:
      return kr_texture_2d_rgb8u_sample_nearest(texture, uv);
      break;
    case KR_TEXTURE_2D_RGBA8U:
      return kr_texture_2d_rgba8u_sample_nearest(texture, uv);
      break;
    default:
      return (kr_vec4) {0};
      break;
  }
  return (kr_vec4) { 0 };
}

kr_vec4
kr_texture_2d_sample(kr_texture* texture, kr_vec2 uv) {
  switch(texture->type) {
    case KR_TEXTURE_2D_RGB8U:
      return kr_texture_2d_rgb8u_sample(texture, uv);
      break;
    case KR_TEXTURE_2D_RGBA8U:
      return kr_texture_2d_rgba8u_sample(texture, uv);
      break;
    default:
      return (kr_vec4) {0};
      break;
  }
  return (kr_vec4) { 0, 0, 1, 0 };
}

kr_internal kr_vec2
kr_sampler_wrap(kr_sampler* sampler, kr_vec2 uv) {
    switch (sampler->wrapS) {
    case KR_WRAP_MODE_REPEAT:
        uv.x = kr_fract(uv.x);
        break;
    default:
        break;
    }

    switch (sampler->wrapT) {
    case KR_WRAP_MODE_REPEAT:
        uv.y = kr_fract(uv.y);
        break;
    default:
        break;
    }
    return uv;
}

kr_vec3
kr_texture_2d_sample3(kr_texture* texture, kr_sampler* sampler, kr_vec2 uv) {
  uv = kr_sampler_wrap(sampler, uv);
  cvec4 xyzw = kr_texture_2d_sample(texture, uv);
  return (cvec3) { xyzw.x, xyzw.y, xyzw.z};
}

kr_vec4
kr_texture_2d_sample4(kr_texture* texture, kr_vec2 uv) {
  return kr_texture_2d_sample(texture, uv);
}

#ifdef KR_HAS_FREEIMAGE
kr_internal kr_texture_type
kr_texture_type_from_free_image_type(FREE_IMAGE_TYPE fit, u32 bpp) {
  switch(fit) {
    case FIT_BITMAP:
      switch(bpp) {
        case 32:
          return KR_TEXTURE_2D_RGBA8U;
          break;
        case 24:
          return KR_TEXTURE_2D_RGB8U;
          break;
        default:
          break;
      }
      break;
    case FIT_RGBAF:
      break;
    case FIT_RGBF:
      break;
    default:
      break;
  }
  return KR_TEXTURE_NONE;
}
#endif

kr_error
kr_texture_save_tiff(kr_texture* texture, const char* filename, b32 flip) {
#ifdef KR_HAS_FREEIMAGE
    FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBAF, (int)texture->dims.x, (int)texture->dims.y, 8 * 4 * sizeof(float), 0, 0, 0);
    if (kr_null == bitmap)
        return (kr_error)"Failed to save file";
    float* bits = (float*)FreeImage_GetBits(bitmap);
    memcpy(bits, texture->data, texture->dims.x * texture->dims.y * 4 * sizeof(float));

    if (flip) FreeImage_FlipVertical(bitmap);
    BOOL done = FreeImage_Save(FIF_TIFF, bitmap, filename, TIFF_NONE);
    FreeImage_Unload(bitmap);

    if (!done)
        return (kr_error)"Failed to save file";
#endif
    return kr_success;
}

kr_error
kr_texture_save_png(kr_texture* texture, const char* filename, b32 flip) {
#ifdef KR_HAS_FREEIMAGE
    FIBITMAP* bitmap = FreeImage_Allocate((int)texture->dims.x, (int)texture->dims.y, 32, 0, 0, 0);
    if (kr_null == bitmap)
        return (kr_error)"Failed to save file";
    u8* bits = (u8*)FreeImage_GetBits(bitmap);
    memcpy(bits, texture->data, texture->dims.x * texture->dims.y * 4 * sizeof(u8));

    if(kr_true == flip) FreeImage_FlipVertical(bitmap);

    BOOL done = FreeImage_Save(FIF_PNG, bitmap, filename, PNG_DEFAULT);
    FreeImage_Unload(bitmap);

    if (!done)
        return (kr_error)"Failed to save file";
#endif    
    return kr_success;
}

kr_error
kr_texture_create_from_file(kr_texture* texture, const char* filename) {
#ifdef KR_HAS_FREEIMAGE
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename, 0);
  FIBITMAP *bitmap = kr_null;
  
  switch (fif) {
      case FIF_JPEG:
        bitmap = FreeImage_Load(FIF_JPEG, filename, JPEG_DEFAULT);
        if(bitmap) FreeImage_FlipVertical(bitmap);
        break;
      case FIF_PNG:
        bitmap = FreeImage_Load(FIF_PNG, filename, PNG_DEFAULT);
        if(bitmap) FreeImage_FlipVertical(bitmap);
        break;
      default:
        return kr_success;
  }

  if(!bitmap) {
    return kr_success;
  }

  u32 image_width = FreeImage_GetWidth(bitmap);
  u32 image_height = FreeImage_GetHeight(bitmap);
  u32 image_bpp = FreeImage_GetBPP(bitmap);
  FREE_IMAGE_TYPE image_type = FreeImage_GetImageType(bitmap);

  BYTE* image_data = FreeImage_GetBits(bitmap);
  
  texture->type = kr_texture_type_from_free_image_type(image_type, image_bpp);
  texture->dims = (uvec3) {image_width, image_height, 0};
  texture->data = kr_texture_allocate(texture->type, texture->dims);
  memcpy(texture->data, image_data, kr_texture_memory_size(texture->type, texture->dims));
  
  FreeImage_Unload(bitmap);
  return kr_success;
#else
  return ((kr_error)"Not supported");
#endif
}
