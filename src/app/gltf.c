#include "common/korangar.h"
#include "common/queue.h"
#include "common/vecmath.h"
#include "common/util.h"
#include "common/logger.h"

#define KR_HAS_FREEIMAGE 1
#include "common/texture.h"

#include <stdio.h>
#include <string.h>

/* These are straight from GL.h */
#define KR_GLTF_BYTE 0x1400
#define KR_GLTF_UNSIGNED_BYTE 0x1401
#define KR_GLTF_SHORT 0x1402
#define KR_GLTF_UNSIGNED_SHORT 0x1403
#define KR_GLTF_INT 0x1404
#define KR_GLTF_UNSIGNED_INT 0x1405
#define KR_GLTF_FLOAT 0x1406

typedef enum
{
    KR_TOKEN_NONE,
    KR_TOKEN_NAME,
    KR_TOKEN_INTEGER,
    KR_TOKEN_FLOAT,
    KR_TOKEN_TRUE,
    KR_TOKEN_FALSE,
    KR_TOKEN_STRING,
    KR_TOKEN_COLON,
    KR_TOKEN_COMMA,
    KR_TOKEN_LBRACKET,
    KR_TOKEN_RBRACKET,
    KR_TOKEN_LBRACE,
    KR_TOKEN_RBRACE,
    KR_TOKEN_POUND,
    KR_TOKEN_DPOUND,
    KR_TOKEN_FORWARD_SLASH,
    KR_TOKEN_ASTERISK,
    KR_TOKEN_EOS,
    KR_TOKEN_MAX
} kr_token_type;

typedef union
{
    double    as_double;
    long long as_integer;
} kr_scalar_u;

typedef struct
{
    const char* view;
    kr_size      length;
    kr_scalar_u  value;
    kr_token_type  type;
} kr_token;

typedef enum
{
    KR_PROPERTY_NONE,
    KR_PROPERTY_VEC2,
    KR_PROPERTY_UVEC2,
    KR_PROPERTY_VEC3,
    KR_PROPERTY_UVEC3,
    KR_PROPERTY_VEC4,
    KR_PROPERTY_UVEC4,
    KR_PROPERTY_ARRAY,
    KR_PROPERTY_MAX
} kr_value_type;

typedef struct
{
    kr_value_type type;
    union {
        kr_vec2  as_vec2;
        kr_uvec2 as_uvec2;
        kr_vec3  as_vec3;
        kr_uvec3 as_uvec3;
        kr_vec4  as_vec4;
        kr_uvec4 as_uvec4;
        kr_mat4  as_mat4;
        struct {
            kr_scalar* data;
            kr_size    count;
        } as_array;
    };
} kr_property_value;

typedef struct
{
    kr_token section;
    kr_token key;
    kr_property_value value;
} kr_property;

kr_internal b32
kr_is_space(char character)
{
    return (character == ' ' || character == '\t' || character == '\v' || character == '\f');
}

kr_internal b32
kr_is_digit(char character)
{
    return character >= '0' && character <= '9';
}

kr_internal int
kr_char_to_digit(char character_value)
{
    switch (character_value) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
        return character_value - '0';
    case 'A':
    case 'a':
        return 10;
    case 'B':
    case 'b':
        return 11;
    case 'C':
    case 'c':
        return 12;
    case 'D':
    case 'd':
        return 13;
    case 'E':
    case 'e':
        return 14;
    case 'F':
    case 'f':
        return 15;
    default:
        return -1;
    }
}

kr_internal kr_error kr_token_next_arithmetic(kr_token* token) {
    kr_scalar_u scalar;
    b32           skip_float = kr_true;

    char const* token_start = token->view;
    char const* token_at = token->view;
    int         base = 10;
    char        at = *token_at++;
    b32         flip_sign = (at == '-');

    if (flip_sign || (at == '+'))
        at = *token_at++;
    while (kr_is_space(at)) {
        at = *token_at++;
    }

    if (at == '0') {
        at = *token_at++;
        if (at == 'x' || at == 'X') {
            at = *token_at++;
            base = 16;
        }
        else if (at == 'b' || at == 'B') {
            at = *token_at++;
            base = 2;
        }
        else if (kr_is_digit(at)) {
            base = 8;
        }
    }
    long long value = 0;
    do {
        int digit = kr_char_to_digit(at);
        if (digit < 0 || digit >= base) {
            break;
        }

        value = value * base + digit;
        at = *token_at++;
    } while (1);
    token_at--;

    at = *token_at;
    scalar.as_integer = value;

    if (at == '.') {
        scalar.as_double = (double)(scalar.as_integer);

        token_at++;
        double pow10 = 0.1;
        double real_value = 0.0;
        while (1) {
            int digit = kr_char_to_digit(*token_at);
            if (digit < 0 || digit > 9) {
                break;
            }
            scalar.as_double += digit * pow10;
            pow10 *= 0.1;
            token_at++;
        }

        // GrbToken fractional_part = lexer.view[0 ..token_end];

        if (*token_at == 'f' || *token_at == 'F') {
            token_at++;
        }

        if ((*token_at == 'e') || (*token_at == 'E')) {
            int    frac = 0;
            double scale = 1.0;

            token_at++;
            if (*token_at == '-' || *token_at == '+') {
                frac = 1;
                token_at++;
            }

            unsigned long long exp_value = 0;
            while (1) {
                int digit = kr_char_to_digit(*token_at);
                if (digit < 0 || digit > 9) {
                    break;
                }
                exp_value = exp_value * 10 + digit;
                token_at++;
            }
            // printf("Exp %d\n", exp_value);
            while (exp_value > 0) {
                scale *= 10.0;
                exp_value -= 1;
            }

            scalar.as_double = (frac ? (scalar.as_double / scale) : (scalar.as_double * scale));
        }
    }
    else {
        token->length = token_at - token_start;
        token->type = KR_TOKEN_INTEGER;
        scalar.as_integer = (kr_true == flip_sign) ? -scalar.as_integer : scalar.as_integer;
        token->value = scalar;

        return kr_null;
    }

    token->length = token_at - token_start;
    token->type = KR_TOKEN_FLOAT;
    scalar.as_double = (kr_true == flip_sign) ? -scalar.as_double : scalar.as_double;
    token->value = scalar;

    return kr_null;
}

kr_internal kr_b32 kr_is_character(char character)
{
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z');
}

kr_internal kr_b32 kr_token_cmp(const char* name, kr_token* token)
{
    return (0 == strncmp(name, token->view, token->length));
}

kr_internal kr_b32 kr_is_id_character(char character)
{
    return character == '_' || kr_is_character(character) || kr_is_digit(character);
}

kr_internal kr_error kr_token_next_name(kr_token* token)
{
    char const* token_at = token->view;

    do {
        if (!kr_is_id_character(*token_at))
            break;
    } while (*token_at++);

    token->length = token_at - token->view;
    token->type = KR_TOKEN_NAME;
    if (kr_token_cmp("false", token)) {
        token->type = KR_TOKEN_FALSE;
    }
    else if (kr_token_cmp("true", token)) {
        token->type = KR_TOKEN_TRUE;
    }

    return kr_null;
}

kr_internal kr_b32 kr_is_new_line(char character)
{
    return (character == '\n' || character == '\r');
}

kr_internal kr_error kr_token_next_string(kr_token* token)
{
    char const* token_at = ++token->view;

    do {
        token_at += (*token_at == '\\') ? 2 : 0;
    } while (*token_at++ != '"');

    do {
        if (!kr_is_id_character(*token_at) || *token_at == '\\')
            break;
    } while (*token_at++);

    token->length = token_at - token->view - 1;
    token->type = KR_TOKEN_STRING;

    return kr_null;
}

kr_internal kr_error kr_token_next(kr_token* token)
{
    kr_error error;
    token->view += token->length;
    token->view += (token->type == KR_TOKEN_STRING) ? 1 : 0;
    while (*token->view) {

        char at = *token->view;
        switch (at) {
        case '\n':
        case '\r':
        case ' ':
        case '\t':
        case '\v':
            at = *(++token->view);
            continue;
        case '#':
            token->length = (token->view[1] == '#') ? 2 : 1;
            token->type = (token->view[1] == '#') ? KR_TOKEN_DPOUND : KR_TOKEN_POUND;
            return kr_null;
        case '-':
        case '+':
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            return kr_token_next_arithmetic(token);
            break;
        case '_':
        case 'a':
        case 'b':
        case 'c':
        case 'd':
        case 'e':
        case 'f':
        case 'g':
        case 'h':
        case 'i':
        case 'j':
        case 'k':
        case 'l':
        case 'm':
        case 'n':
        case 'o':
        case 'p':
        case 'q':
        case 'r':
        case 's':
        case 't':
        case 'u':
        case 'v':
        case 'w':
        case 'x':
        case 'y':
        case 'z':
        case 'A':
        case 'B':
        case 'C':
        case 'D':
        case 'E':
        case 'F':
        case 'G':
        case 'H':
        case 'I':
        case 'J':
        case 'K':
        case 'L':
        case 'M':
        case 'N':
        case 'O':
        case 'P':
        case 'Q':
        case 'R':
        case 'S':
        case 'T':
        case 'U':
        case 'V':
        case 'W':
        case 'X':
        case 'Y':
        case 'Z':
            error = kr_token_next_name(token);
            if (kr_null != error) {
                while (!kr_is_new_line(*(++token->view))) {
                }

                continue;
            }
            return error;
        case '"':
            return kr_token_next_string(token);
        case ':':
            token->length = 1;
            token->type = KR_TOKEN_COLON;
            return kr_null;
        case ',':
            token->length = 1;
            token->type = KR_TOKEN_COMMA;
            return kr_null;
        case '{':
            token->length = 1;
            token->type = KR_TOKEN_LBRACE;
            return kr_null;
        case '}':
            token->length = 1;
            token->type = KR_TOKEN_RBRACE;
            return kr_null;
        case '[':
            token->length = 1;
            token->type = KR_TOKEN_LBRACKET;
            return kr_null;
        case ']':
            token->length = 1;
            token->type = KR_TOKEN_RBRACKET;
            return kr_null;
        case '/':
            token->length = 1;
            token->type = KR_TOKEN_FORWARD_SLASH;
            return kr_null;
        case '*':
            token->length = 1;
            token->type = KR_TOKEN_ASTERISK;
            return kr_null;
        default:
            token->length = 0;
            return kr_null;
        }
    }

    token->length = 0;
    token->type = KR_TOKEN_EOS;

    return kr_null;
}

kr_internal kr_scalar kr_token_as_scalar(kr_token* token) {
    return (kr_scalar)((KR_TOKEN_INTEGER == token->type) ? token->value.as_integer : token->value.as_double);
}

kr_internal void kr_token_print(kr_token* token) {
    printf("%.*s\n", (i32)token->length, token->view);
}

typedef struct
{
    kr_token uri;
    kr_size byteLength;
    u8* data;
} kr_gltf_buffer;

typedef struct
{
    kr_size buffer;
    kr_size byteLength;
    kr_size byteOffset;
} kr_gltf_buffer_view;

typedef struct
{
    kr_token type;
    kr_size bufferView;
    kr_size componentType;
    kr_size count;
} kr_gltf_accessor;

typedef struct
{
    i32 positions;
    i32 normals;
    i32 uvs;
    i32 indices;
    i32 material;
} kr_gltf_mesh_primitive;
kr_internal const kr_gltf_mesh_primitive g_gltf_default_mesh_primitive = { -1, -1, -1, -1, -1 };

typedef struct
{
    kr_transform transform;
    kr_token name;
    i32  parent;
    i32* children;
    i32  mesh;
    i32  camera;
} kr_gltf_node;

typedef struct
{
    kr_token mime;
    kr_token name;
    kr_token uri;
} kr_gltf_image;

typedef struct {
    kr_i32 wrapS;
    kr_i32 wrapT;
} kr_gltf_sampler;

typedef struct
{
    kr_i32 source;
    kr_i32 sampler;
} kr_gltf_texture;

typedef struct
{
    kr_token name;
    kr_ivec2 base_color_texture;
    kr_vec3 base_color;
    kr_scalar roughness;
    kr_scalar metalicity;
    kr_i32 base_color_sampler;
} kr_gltf_material;

typedef struct
{
    kr_token name;
    kr_gltf_mesh_primitive* primitives;
} kr_gltf_mesh;

typedef struct
{
    kr_token name;
    kr_scalar ar;
    kr_scalar fov;
    kr_camera_type type;
    i32 node;
} kr_gltf_camera;

typedef struct
{
    kr_gltf_buffer* buffers;
    kr_gltf_buffer_view* buffer_views;
    kr_gltf_accessor* accessors;
    kr_gltf_camera* cameras;
    kr_gltf_mesh* meshes;
    kr_gltf_node* nodes;
    kr_gltf_image* images;
    kr_gltf_texture* textures;
    kr_gltf_sampler* samplers;
    kr_gltf_material* materials;
} kr_gltf_object;

kr_internal kr_error
kr_geometry_gltf_parse_object(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_buffer(kr_gltf_object* gltf_object, kr_gltf_buffer* gltf_buffer, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_buffer_view(kr_gltf_object* gltf_object, kr_gltf_buffer_view* gltf_buffer_view, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_accessor(kr_gltf_object* gltf_object, kr_gltf_accessor* gltf_accessor, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_primitive(kr_gltf_object* gltf_object, kr_gltf_mesh_primitive* gltf_mesh_primitive, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_meshes(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_material(kr_gltf_object* gltf_object, kr_gltf_material* material, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_materials(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_texture(kr_gltf_object* gltf_object, kr_gltf_texture* texture, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_textures(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_samplers(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_cameras(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_sampler(kr_gltf_object* gltf_object, kr_gltf_sampler* sampler, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_image(kr_gltf_object* gltf_object, kr_gltf_image* image, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_images(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_nodes(kr_gltf_object* gltf_object, kr_token* token);
kr_internal kr_error
kr_geometry_gltf_parse_array(kr_gltf_object* gltf_object, kr_token* token);

kr_error
kr_geometry_gltf_parse_accessor(kr_gltf_object* gltf_object, kr_gltf_accessor* gltf_accessor, kr_token* token) {
    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_LBRACKET:
            kr_geometry_gltf_parse_array(gltf_object, token);
            break;
        default:
            if (kr_true == kr_token_cmp("bufferView", &key)) {
                gltf_accessor->bufferView = value.value.as_integer;
            }
            else if (kr_true == kr_token_cmp("componentType", &key)) {
                gltf_accessor->componentType = value.value.as_integer;
            }
            else if (kr_true == kr_token_cmp("count", &key)) {
                gltf_accessor->count = value.value.as_integer;
            }
            else if (kr_true == kr_token_cmp("type", &key)) {
                gltf_accessor->type = value;
            }
            else {

            }
            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_buffer_view(kr_gltf_object* gltf_object, kr_gltf_buffer_view* gltf_buffer_view, kr_token* token) {
    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        if (kr_true == kr_token_cmp("byteLength", &key)) {
            gltf_buffer_view->byteLength = value.value.as_integer;
        }
        else if (kr_true == kr_token_cmp("byteOffset", &key)) {
            gltf_buffer_view->byteOffset = value.value.as_integer;
        }
        else if (kr_true == kr_token_cmp("buffer", &key)) {
            gltf_buffer_view->buffer = value.value.as_integer;
        }
        kr_token_next(token);
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_buffer(kr_gltf_object* gltf_object, kr_gltf_buffer* gltf_buffer, kr_token* token) {

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        if (kr_true == kr_token_cmp("byteLength", &key)) {
            gltf_buffer->byteLength = value.value.as_integer;
        } else if (kr_true == kr_token_cmp("uri", &key)) {
            gltf_buffer->uri = value;
        }
        kr_token_next(token);
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_primitive_attributes(kr_gltf_object* gltf_object, i32* attributes, i32 max_count, kr_token* token) {

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        if (kr_true == kr_token_cmp("POSITION", &key)) {
            attributes[0] = (i32)value.value.as_integer;
        }
        else if (kr_true == kr_token_cmp("NORMAL", &key)) {
            attributes[1] = (i32)value.value.as_integer;
        }
        else if (kr_true == kr_token_cmp("TEXCOORD_0", &key)) {
            attributes[2] = (i32)value.value.as_integer;
        }

        kr_token_next(token);
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_primitive(kr_gltf_object* gltf_object, kr_gltf_mesh_primitive* gltf_mesh_primitive, kr_token* token) {
    i32 attrs[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        if (kr_true == kr_token_cmp("attributes", &key)) {
            kr_geometry_gltf_parse_primitive_attributes(gltf_object, attrs, 8, token);
            gltf_mesh_primitive->positions = attrs[0];
            gltf_mesh_primitive->normals = attrs[1];
            gltf_mesh_primitive->uvs = attrs[2];
        }
        else if (kr_true == kr_token_cmp("indices", &key)) {
            gltf_mesh_primitive->indices = (i32)value.value.as_integer;
            kr_token_next(token);
            //kr_geometry_gltf_parse_array(gltf_object, token);
        } else if (kr_true == kr_token_cmp("material", &key)) {
            gltf_mesh_primitive->material = (i32)value.value.as_integer;
            kr_token_next(token);
        } else {
            if (KR_TOKEN_LBRACKET == token->type) {
                kr_geometry_gltf_parse_array(gltf_object, token);
            }
            else if (KR_TOKEN_LBRACE == token->type) {
                kr_geometry_gltf_parse_object(gltf_object, token);
            }
            else {
                kr_token_next(token);
            }
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_primitives(kr_gltf_object* gltf_object, kr_gltf_mesh* gltf_mesh, kr_token* token) {
    kr_gltf_mesh_primitive gltf_mesh_primitive = g_gltf_default_mesh_primitive;

    kr_queue_init(gltf_mesh->primitives, 1);

    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_primitive(gltf_object, &gltf_mesh_primitive, token);
            kr_queue_push(gltf_mesh->primitives, gltf_mesh_primitive);
            break;
        case KR_TOKEN_LBRACKET:
            /*if (kr_true == kr_token_cmp("primitives", &key)) {
                kr_geometry_gltf_parse_primitive(gltf_object, &gltf_mesh_primitive, 8, token);
            }
            else {
                kr_geometry_gltf_parse_array(gltf_object, token);
            }*/
            break;
        default:
            //if (kr_true == kr_token_cmp("name", &key)) {
            //}
            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_mesh(kr_gltf_object* gltf_object, kr_gltf_mesh* gltf_mesh, kr_token* token) {
    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_LBRACKET:
            if (kr_true == kr_token_cmp("primitives", &key)) {
                kr_geometry_gltf_parse_primitives(gltf_object, gltf_mesh, token);
            } else {
                kr_geometry_gltf_parse_array(gltf_object, token);
            }
            break;
        default:
            if (kr_true == kr_token_cmp("name", &key)) {
                gltf_mesh->name = value;
            }
            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_vec3(kr_gltf_object* gltf_object, vec3* vector, kr_token* token) {
    kr_token_next(token);
    vector->x = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);
    vector->y = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);
    vector->z = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_mat4(kr_gltf_object* gltf_object, mat4* matrix, kr_token* token) {
    kr_token_next(token);
    for (int i = 0; i < 16; i++) {
        matrix->v[i] = kr_token_as_scalar(token);
        kr_token_next(token);
        kr_token_next(token);
    }
   
    return kr_success;
}
kr_error
kr_geometry_gltf_parse_vec4(kr_gltf_object* gltf_object, vec4* vector, kr_token* token) {
    kr_token_next(token);
    vector->x = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);
    vector->y = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);
    vector->z = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);
    vector->w = kr_token_as_scalar(token);
    kr_token_next(token);
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_node(kr_gltf_object* gltf_object, kr_gltf_node* gltf_node, kr_token* token) {
    mat4 matrix = kr_midentity4();
    vec4 rotation = { 0,0,0,1 };
    vec3 translation = { 0 };
    vec3 scale = { 1, 1, 1 };
    b32 use_matrix = kr_false;

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        switch (token->type) {
        case KR_TOKEN_LBRACKET: {
            if (kr_true == kr_token_cmp("rotation", &key)) {
                kr_geometry_gltf_parse_vec4(gltf_object, &rotation, token);
            } else if (kr_true == kr_token_cmp("translation", &key)) {
                kr_geometry_gltf_parse_vec3(gltf_object, &translation, token);
            } else if (kr_true == kr_token_cmp("scale", &key)) {
                kr_geometry_gltf_parse_vec3(gltf_object, &scale, token);
            }
            else if (kr_true == kr_token_cmp("matrix", &key)) {
                kr_geometry_gltf_parse_mat4(gltf_object, &matrix, token);
                use_matrix = kr_true;
            }
            else if (kr_true == kr_token_cmp("children", &key)) {
                kr_queue_init(gltf_node->children, 1);
                while (KR_TOKEN_RBRACKET != token->type) {
                    kr_token_next(token);
                    kr_queue_push(gltf_node->children, token->value.as_integer);
                    kr_token_next(token);
                }
                kr_token_next(token);

            }
            else {
                kr_geometry_gltf_parse_array(gltf_object, token);
            }
        } break;
        default:
            if (kr_true == kr_token_cmp("mesh", &key)) {
                gltf_node->mesh = (i32)value.value.as_integer;
            }
            else if (kr_true == kr_token_cmp("camera", &key)) {
                gltf_node->camera = (i32)value.value.as_integer;
            }
            else if (kr_true == kr_token_cmp("name", &key)) {
                gltf_node->name = value;
            }
            kr_token_next(token);
            break;
        }
    }

    mat4 mrotate = kr_mquat4(rotation);
    mat4 mtranslate = kr_mtranslate4(translation);
    mat4 mscale = kr_mscale4(scale);

    gltf_node->transform = (kr_true == use_matrix) ? kr_mtransform4(matrix) : kr_mtransform4(kr_mmul4(mtranslate, kr_mmul4(mrotate, mscale)));
    //gltf_node->transform = kr_mtransform4(mtranslate);

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_nodes(kr_gltf_object* gltf_object, kr_token* token) {
    
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_node gltf_node = { 0 };
            gltf_node.mesh = -1;
            gltf_node.camera = -1;
            gltf_node.parent = -1;
            kr_geometry_gltf_parse_node(gltf_object, &gltf_node, token);
            kr_queue_push(gltf_object->nodes, gltf_node);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_base_color_texture_info(kr_gltf_object* gltf_object, kr_gltf_material* material, kr_token* token) {
    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        
        if (kr_true == kr_token_cmp("index", &key)) {
          material->base_color_texture.x = (i32)value.value.as_integer;
        } else if (kr_true == kr_token_cmp("texCoord", &key)) {
          material->base_color_texture.y = (i32)value.value.as_integer;
        }
        
        kr_token_next(token);
    }
    
    kr_token_next(token);
	return kr_success;
}

kr_error
kr_geometry_gltf_parse_pbr_material(kr_gltf_object* gltf_object, kr_gltf_material* material, kr_token* token) {
while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        
        switch (token->type) {
        case KR_TOKEN_LBRACE:
            if (kr_true == kr_token_cmp("baseColorTexture", &key)) {
              kr_geometry_gltf_parse_base_color_texture_info(gltf_object, material, token);
            } else {
              kr_geometry_gltf_parse_object(gltf_object, token);
            }
            break;
        case KR_TOKEN_LBRACKET:
            if (kr_true == kr_token_cmp("baseColorFactor", &key)) {
               vec4 base_color = {0};
               kr_geometry_gltf_parse_vec4(gltf_object, &base_color, token);
               material->base_color = (vec3) {base_color.x, base_color.y, base_color.z};
            } else {
                kr_geometry_gltf_parse_array(gltf_object, token);
            }
            break;
        default:
           if (kr_true == kr_token_cmp("roughnessFactor", &key)) {
               material->roughness = kr_token_as_scalar (&value);
           } else if (kr_true == kr_token_cmp("metallicFactor", &key)) {
               material->metalicity = kr_token_as_scalar (&value);
           }
            kr_token_next(token);
            break;
        }
    }
    
    kr_token_next(token);

	return kr_success;
}
 
kr_error
kr_geometry_gltf_parse_material(kr_gltf_object* gltf_object, kr_gltf_material* material, kr_token* token) {
 
 while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        
        switch (token->type) {
        case KR_TOKEN_LBRACE:
           if (kr_true == kr_token_cmp("pbrMetallicRoughness", &key)) {
               kr_geometry_gltf_parse_pbr_material(gltf_object, material, token);
           } else {
            kr_geometry_gltf_parse_object(gltf_object, token);
           }
            break;
        case KR_TOKEN_LBRACKET:
            kr_geometry_gltf_parse_array(gltf_object, token);
            break;
        default:
                        //kr_token_print(&key);

            if (kr_true == kr_token_cmp("name", &key)) {
              material->name = value;
            } else if (kr_true == kr_token_cmp("pbrMetallicRoughness", &key)) {
                
            }

            kr_token_next(token);
            break;
        }
    }
    
    kr_token_next(token);
	return kr_success;
}

kr_error
kr_geometry_gltf_parse_materials(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_material gltf_material = { 0 };
            gltf_material.base_color_texture = (ivec2) { -1, -1 };

            kr_geometry_gltf_parse_material(gltf_object, &gltf_material, token);
            kr_queue_push(gltf_object->materials, gltf_material);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_texture(kr_gltf_object* gltf_object, kr_gltf_texture* texture, kr_token* token) {

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        
        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_LBRACKET:
            break;
        default:
            if (kr_true == kr_token_cmp("source", &key)) {
                texture->source = (i32)value.value.as_integer;
            } else if (kr_true == kr_token_cmp("sampler", &key)) {
                texture->sampler = (i32)value.value.as_integer;
            }
            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_textures(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_texture gltf_texture = { -1, -1 };
            kr_geometry_gltf_parse_texture(gltf_object, &gltf_texture, token);
            kr_queue_push(gltf_object->textures, gltf_texture);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_internal kr_error
kr_geometry_gltf_parse_sampler(kr_gltf_object* gltf_object, kr_gltf_sampler* sampler, kr_token* token) {

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);

        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;

        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_LBRACKET:
            break;
        default:
            if (kr_true == kr_token_cmp("magFilter", &key)) {

            } else if (kr_true == kr_token_cmp("minFilter", &key)) {

            } else if (kr_true == kr_token_cmp("wrapT", &key)) {
                sampler->wrapT = KR_WRAP_MODE_REPEAT;
            } else if (kr_true == kr_token_cmp("wrapS", &key)) {
                sampler->wrapS = KR_WRAP_MODE_REPEAT;
            }
            
            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);
}


kr_internal kr_error
kr_geometry_gltf_parse_perspective_camera(kr_gltf_object* gltf_object, kr_gltf_camera* camera, kr_token* token) {
    camera->ar = 1.0f;
    camera->fov = kr_radians(60.0f);

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);

        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;

        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_LBRACKET:
            break;
        default:
            if (kr_true == kr_token_cmp("aspectRatio", &key)) {
                camera->ar = kr_token_as_scalar(&value);
            }
            else if (kr_true == kr_token_cmp("yfov", &key)) {
                camera->fov = kr_token_as_scalar(&value);
            }
            else if (kr_true == kr_token_cmp("zfar", &key)) {
            }
            else if (kr_true == kr_token_cmp("znear", &key)) {
            }

            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);
}

kr_internal kr_error
kr_geometry_gltf_parse_camera(kr_gltf_object* gltf_object, kr_gltf_camera* camera, kr_token* token) {
    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);

        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;

        switch (token->type) {
        case KR_TOKEN_LBRACE:
            if (kr_true == kr_token_cmp("perspective", &key)) {
                kr_geometry_gltf_parse_perspective_camera(gltf_object, camera, token);
            }            
            break;
        case KR_TOKEN_LBRACKET:
            break;
        default:
            if (kr_true == kr_token_cmp("name", &key)) {
                camera->name = value;
            }
            else if (kr_true == kr_token_cmp("type", &key)) {
                camera->type = KR_CAMERA_PINHOLE;
            }

            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_internal kr_error
kr_geometry_gltf_parse_cameras(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);

        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_camera gltf_camera = { 0 };
            gltf_camera.node = -1;
            kr_geometry_gltf_parse_camera(gltf_object, &gltf_camera, token);
            kr_queue_push(gltf_object->cameras, gltf_camera);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_internal kr_error
kr_geometry_gltf_parse_samplers(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);

        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_sampler gltf_sampler = { 0 };
            kr_geometry_gltf_parse_sampler(gltf_object, &gltf_sampler, token);
            kr_queue_push(gltf_object->samplers, gltf_sampler);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_image(kr_gltf_object* gltf_object, kr_gltf_image* image, kr_token* token) {
    
    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        
        kr_token key = *token;
        kr_token_next(token);
        kr_token_next(token);
        kr_token value = *token;
        
        switch (token->type) {
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_LBRACKET:
            break;
        default:
            if (kr_true == kr_token_cmp("mimeType", &key)) {
                image->mime = value;
            } else if (kr_true == kr_token_cmp("name", &key)) {
                image->name = value;
            } else if (kr_true == kr_token_cmp("uri", &key)) {
                image->uri = value;
            }
            kr_token_next(token);
            break;
        }
    }
    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_images(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_image gltf_image = { 0 };
            kr_geometry_gltf_parse_image(gltf_object, &gltf_image, token);
            kr_queue_push(gltf_object->images, gltf_image);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_meshes(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_mesh gltf_mesh = { 0 };
            kr_geometry_gltf_parse_mesh(gltf_object, &gltf_mesh, token);
            kr_queue_push(gltf_object->meshes, gltf_mesh);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_accessors(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_accessor gltf_accessor = { 0 };
            kr_geometry_gltf_parse_accessor(gltf_object, &gltf_accessor, token);
            kr_queue_push(gltf_object->accessors, gltf_accessor);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_buffers(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_buffer gltf_buffer = { 0 };
            kr_geometry_gltf_parse_buffer(gltf_object, &gltf_buffer, token);
            kr_queue_push(gltf_object->buffers, gltf_buffer);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}


kr_error
kr_geometry_gltf_parse_buffer_views(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACE: {
            kr_gltf_buffer_view gltf_buffer_view = { 0 };
            kr_geometry_gltf_parse_buffer_view(gltf_object, &gltf_buffer_view, token);
            kr_queue_push(gltf_object->buffer_views, gltf_buffer_view);
            break;
        }
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_array(kr_gltf_object* gltf_object, kr_token* token) {
    while (KR_TOKEN_RBRACKET != token->type) {
        kr_token_next(token);
        switch (token->type) {
        case KR_TOKEN_LBRACKET:
            kr_geometry_gltf_parse_array(gltf_object, token);
            break;
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        }
    }

    kr_token_next(token);

    return kr_success;
}

kr_error
kr_geometry_gltf_parse_object(kr_gltf_object* gltf_object, kr_token* token) {
    kr_token key = { 0 };

    while (KR_TOKEN_RBRACE != token->type) {
        kr_token_next(token);
        /* Handle empty object */
        if (KR_TOKEN_RBRACE == token->type) break;

        key = *token;

        kr_token_next(token);
        // assert ':'

        kr_token_next(token);

        switch (token->type) {
        case KR_TOKEN_LBRACKET:
            if (kr_true == kr_token_cmp("buffers", &key)) {
                kr_geometry_gltf_parse_buffers(gltf_object, token);
            } else if (kr_true == kr_token_cmp("bufferViews", &key)) {
                kr_geometry_gltf_parse_buffer_views(gltf_object, token);
            } else if (kr_true == kr_token_cmp("accessors", &key)) {
                kr_geometry_gltf_parse_accessors(gltf_object, token);
            } else if (kr_true == kr_token_cmp("meshes", &key)) {
                kr_geometry_gltf_parse_meshes(gltf_object, token);
            } else if (kr_true == kr_token_cmp("images", &key)) {
                kr_geometry_gltf_parse_images(gltf_object, token);
            } else if (kr_true == kr_token_cmp("textures", &key)) {
                kr_geometry_gltf_parse_textures(gltf_object, token);
            } else if (kr_true == kr_token_cmp("samplers", &key)) {
                kr_geometry_gltf_parse_samplers(gltf_object, token);
            } else if (kr_true == kr_token_cmp("materials", &key)) {
                //kr_geometry_gltf_parse_materials(gltf_object, token);
                kr_geometry_gltf_parse_array(gltf_object, token);
            } else if (kr_true == kr_token_cmp("nodes", &key)) {
                kr_geometry_gltf_parse_nodes(gltf_object, token);
            } else if (kr_true == kr_token_cmp("cameras", &key)) {
                kr_geometry_gltf_parse_cameras(gltf_object, token);
            } else {
                kr_geometry_gltf_parse_array(gltf_object, token);
            }

            break;
        case KR_TOKEN_LBRACE:
            kr_geometry_gltf_parse_object(gltf_object, token);
            break;
        case KR_TOKEN_COMMA:
            kr_token_next(token);
            break;
        default:
            kr_token_next(token);
            break;
        } 
    }

    kr_token_next(token);

    return kr_success;
}



kr_error 
kr_geometry_gltf_parse(kr_gltf_object* gltf_object, const char* content, const char* base_path) {
    kr_error result = kr_success;

    kr_queue_init(gltf_object->buffers, 1);
    kr_queue_init(gltf_object->buffer_views, 1);
    kr_queue_init(gltf_object->accessors, 1);
    kr_queue_init(gltf_object->meshes, 1);
    kr_queue_init(gltf_object->nodes, 1);
    kr_queue_init(gltf_object->images, 1);
    kr_queue_init(gltf_object->textures, 1);
    kr_queue_init(gltf_object->materials, 1);
    kr_queue_init(gltf_object->samplers, 1);
    kr_queue_init(gltf_object->cameras, 1);
    
    kr_token token = { content, 0, KR_TOKEN_NONE };

    kr_property* properties = kr_null;
    kr_queue_init(properties, 1);

    kr_token_next(&token);

    kr_geometry_gltf_parse_object(gltf_object, &token);

    kr_size node_count = kr_queue_size(gltf_object->nodes);
    for (kr_size node_index = 0; node_index < node_count; node_index++) {
        kr_gltf_node* node = &gltf_object->nodes[node_index];
        if (!node->children) continue;
        kr_size child_count = kr_queue_size(node->children);
        for (int child_index = 0; child_index < child_count; child_index++) {
            gltf_object->nodes[node->children[child_index]].parent = node_index;
        }
        printf("\n");
    }

    for (kr_size node_index = 0; node_index < node_count; node_index++) {
        kr_gltf_node* node = &gltf_object->nodes[node_index];
    }

    for (kr_size node_index = 0; node_index < node_count; node_index++) {
        kr_gltf_node* node = &gltf_object->nodes[node_index];
        if (node->camera < 0) continue;
        kr_gltf_camera* camera = &gltf_object->cameras[node->camera];
        camera->node = (i32)node_index;
        //printf("Camera %.*s\n", (i32)camera->name.length, camera->name.view);
    }

    kr_size camera_count = kr_queue_size(gltf_object->cameras);
    for (kr_size camera_index = 0; camera_index < camera_count; camera_index++) {
        kr_gltf_camera* camera = &gltf_object->cameras[camera_index];
        //printf("Camera %.*s\n", (i32)camera->name.length, camera->name.view);
    }

    kr_size buffer_count = kr_queue_size(gltf_object->buffers);
    for (kr_size buffer_index = 0; buffer_index < buffer_count; buffer_index++) {
        kr_gltf_buffer* buffer = &gltf_object->buffers[buffer_index];
        if (KR_TOKEN_NONE == buffer->uri.type) {
            continue;
        }

        char uri[256] = { 0 };
        sprintf(uri, "%s/%.*s", base_path, (i32)buffer->uri.length, buffer->uri.view);

        FILE* bin = fopen(uri, "rb");
        if (!bin) {
            continue;
        }

        buffer->data = kr_allocate(buffer->byteLength);
        fread(buffer->data, 1, buffer->byteLength, bin);
        fclose(bin);
    }

    kr_queue_release(properties);

    return kr_success;
}

kr_internal kr_error 
kr_object_gltf_create(kr_object* object, kr_object_instance* instance, kr_gltf_object* gltf_object, kr_size mesh_index) {
    u32 face_count = 0;
    u32 attr_count = 0;

    kr_gltf_mesh* gltf_mesh = &gltf_object->meshes[mesh_index];
    kr_size primitive_count = kr_queue_size(gltf_mesh->primitives);
    
    for (kr_size primitive_index = 0; primitive_index < primitive_count; primitive_index++) {
        kr_gltf_mesh_primitive* primitive = &gltf_mesh->primitives[primitive_index];

        if (primitive->indices >= 0) {
            kr_gltf_accessor* accessor = &gltf_object->accessors[primitive->indices];

            face_count += (u32)accessor->count / 3;
        }

        if (primitive->positions >= 0) {
            kr_gltf_accessor* accessor = &gltf_object->accessors[primitive->positions];

            attr_count += (u32)accessor->count;
        }
    }

    object->aabb = kr_aabb_empty3();

    object->type = KR_OBJECT_MESH;
    object->as_mesh.attr_count = attr_count;
    object->as_mesh.face_count = face_count;

    object->as_mesh.faces = kr_aligned_allocate(object->as_mesh.face_count * sizeof(*object->as_mesh.faces), kr_align_of(kr_vec3));
    object->as_mesh.vertices = kr_aligned_allocate(object->as_mesh.attr_count * sizeof(*object->as_mesh.vertices), kr_align_of(kr_vec3));
    object->as_mesh.normals = kr_aligned_allocate(object->as_mesh.attr_count * sizeof(*object->as_mesh.normals), kr_align_of(kr_vec3));
    object->as_mesh.uvs = kr_aligned_allocate(object->as_mesh.attr_count * sizeof(*object->as_mesh.uvs), kr_align_of(kr_vec3));
  
    u32 face_index = 0;
	u32 mesh_offset = 0;
	u32 node_count = (u32)kr_queue_size(gltf_object->nodes);
    kr_gltf_node* node = kr_null;
    for (kr_size node_index = 0; node_index < node_count; node_index++) {
        kr_gltf_node* at = &gltf_object->nodes[node_index];
        if (at->mesh == mesh_index) {
            node = at;
			if(instance)
				instance->model = kr_invtransform(node->transform);
			break;
        }
    }

    for (kr_size primitive_index = 0; primitive_index < primitive_count; primitive_index++) {
        kr_gltf_mesh_primitive* primitive = &gltf_mesh->primitives[primitive_index];
        if (primitive->indices >= 0) {
            kr_gltf_accessor* accessor = &gltf_object->accessors[primitive->indices];
            kr_gltf_buffer_view* buffer_view = &gltf_object->buffer_views[accessor->bufferView];
            kr_gltf_buffer* buffer = &gltf_object->buffers[buffer_view->buffer];
            u32 material_index = (primitive->material < 0) ? 0 : primitive->material;
           
            switch (accessor->componentType) {
            case KR_GLTF_UNSIGNED_SHORT: {
                kr_size index_count = buffer_view->byteLength / 2;
                u16* indices = (u16*)kr_allocate(buffer_view->byteLength);
                memcpy(indices, buffer->data + buffer_view->byteOffset, buffer_view->byteLength);

                for (kr_size index = 0; index < index_count; index+=3) {
                    uvec4 face = { indices[index + 0] + mesh_offset, indices[index + 1] + mesh_offset, indices[index + 2] + mesh_offset, material_index };
                    object->as_mesh.faces[face_index++] = face;
                }
                kr_free(&indices);
            } break;
            case KR_GLTF_UNSIGNED_INT: {
                kr_size index_count = buffer_view->byteLength / 4;
                u32* indices = (u32*)kr_allocate(buffer_view->byteLength);
                memcpy(indices, buffer->data + buffer_view->byteOffset, buffer_view->byteLength);

                for (kr_size index = 0; index < index_count; index += 3) {
                    uvec4 face = { indices[index + 0] + mesh_offset, indices[index + 1] + mesh_offset, indices[index + 2] + mesh_offset, material_index };
                    object->as_mesh.faces[face_index++] = face;
                }
                kr_free(&indices);
            } break;
            default:
                break;
            }
        }

        if (primitive->positions >= 0) {
            kr_gltf_accessor* accessor = &gltf_object->accessors[primitive->positions];
            kr_gltf_buffer_view* buffer_view = &gltf_object->buffer_views[accessor->bufferView];
            kr_gltf_buffer* buffer = &gltf_object->buffers[buffer_view->buffer];

            memcpy(object->as_mesh.vertices + mesh_offset, buffer->data + buffer_view->byteOffset, buffer_view->byteLength);
        }
        if (primitive->uvs >= 0) {
            kr_gltf_accessor* accessor = &gltf_object->accessors[primitive->uvs];
            kr_gltf_buffer_view* buffer_view = &gltf_object->buffer_views[accessor->bufferView];
            kr_gltf_buffer* buffer = &gltf_object->buffers[buffer_view->buffer];

            memcpy(object->as_mesh.uvs + mesh_offset, buffer->data + buffer_view->byteOffset, buffer_view->byteLength);
        }
        if (primitive->normals >= 0) {
            kr_gltf_accessor* accessor = &gltf_object->accessors[primitive->normals];
            kr_gltf_buffer_view* buffer_view = &gltf_object->buffer_views[accessor->bufferView];
            kr_gltf_buffer* buffer = &gltf_object->buffers[buffer_view->buffer];

            memcpy(object->as_mesh.normals + mesh_offset, buffer->data + buffer_view->byteOffset, buffer_view->byteLength);

            mesh_offset += (u32)accessor->count;
        }
    }

    for (kr_size attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
        object->aabb = kr_aabb_expand3(object->aabb, object->as_mesh.vertices[attr_index]);
    }

    kr_log("Mesh '%.*s' face count: %d attr_count: %d\n", gltf_mesh->name.length, gltf_mesh->name.view, face_count, attr_count);
    kr_log("Mesh '%.*s' AABB [%f %f %f] x [%f %f %f]\n", gltf_mesh->name.length, gltf_mesh->name.view, object->aabb.min.x,  object->aabb.min.y, object->aabb.min.z, object->aabb.max.x, object->aabb.max.y,  object->aabb.max.z);

  return kr_success;
}

kr_error
kr_geometry_gltf_create(kr_object* object, const char* file_path, const char* base_path) {
	kr_error result = kr_success;

	char file_path_norm[256] = { 0 };
	kr_path_normalize(file_path, file_path_norm, sizeof(file_path_norm));

	char directory_path[256] = { 0 };
	kr_file_base(file_path_norm, directory_path, sizeof(directory_path));

	base_path = (base_path != kr_null) ? base_path : &directory_path[0];

	kr_size content_size = 0;
	result = kr_file_size(file_path_norm, &content_size);
	if (kr_success != result) return result;

	char* content_buffer = (char*)kr_allocate(content_size + 1);
	content_buffer[content_size] = 0;
	result = kr_file_contents(file_path_norm, content_buffer, content_size);

	kr_gltf_object gltf_object = { 0 };
	result = kr_geometry_gltf_parse(&gltf_object, content_buffer, directory_path);
	if (kr_success != result) return result;

	u32 face_count = 0;
	u32 attr_count = 0;

	kr_size mesh_index = 0;
	kr_size mesh_count = kr_queue_size(gltf_object.meshes);

	kr_object* objects = kr_null;
	kr_queue_init(objects, 1);

	for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		kr_object   object = { 0 };

		kr_object_gltf_create(&object, kr_null, &gltf_object, mesh_index);

		face_count += (u32)object.as_mesh.face_count;
		attr_count += (u32)object.as_mesh.attr_count;

		kr_queue_push(objects, object);
	}

	object->type = KR_OBJECT_MESH;
	object->as_mesh.face_count = face_count;
	object->as_mesh.attr_count = attr_count;

	kr_queue_reserve(object->as_mesh.faces, object->as_mesh.face_count);
	kr_queue_reserve(object->as_mesh.vertices, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.normals, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.uvs, object->as_mesh.attr_count);

	u32 face_index = 0;
	u32 attr_offset = 0;
	u32 face_offset = 0;

	for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		kr_object*   src_object = &objects[mesh_index];

		uvec4* faces = object->as_mesh.faces + face_offset;
		vec3* vertices = object->as_mesh.vertices + attr_offset;
		vec3* normals = object->as_mesh.normals + attr_offset;
		vec2* uvs = object->as_mesh.uvs + attr_offset;

		for (kr_size face_index = 0; face_index < src_object->as_mesh.face_count; face_index++) {
			uvec4 face = src_object->as_mesh.faces[face_index];
			faces[face_index] = (uvec4) {
				face.x + attr_offset,
					face.y + attr_offset,
					face.z + attr_offset,
					face.w
			};
		}

		kr_memcpy(vertices, src_object->as_mesh.vertices, src_object->as_mesh.attr_count * sizeof(*vertices));
		kr_memcpy(normals, src_object->as_mesh.normals, src_object->as_mesh.attr_count * sizeof(*normals));
		kr_memcpy(uvs, src_object->as_mesh.uvs, src_object->as_mesh.attr_count * sizeof(*uvs));

		object->aabb = kr_aabb_expand(object->aabb, src_object->aabb);

		face_offset += (u32)src_object->as_mesh.face_count;
		attr_offset += (u32)src_object->as_mesh.attr_count;
	}

	return kr_success;
}

kr_internal kr_error
kr_gltf_object_destroy(kr_gltf_object* gltf_object) {
    kr_size mesh_index = 0;
    kr_size mesh_count = kr_queue_size(gltf_object->meshes);
    kr_size buffer_count = kr_queue_size(gltf_object->buffers);
    for (kr_size buffer_index = 0; buffer_index < buffer_count; buffer_index++) {
        kr_gltf_buffer* buffer = &gltf_object->buffers[buffer_index];
        kr_free(&buffer->data);
    }

    for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
        kr_gltf_mesh* gltf_mesh = &gltf_object->meshes[mesh_index];
        kr_queue_release(gltf_mesh->primitives);
    }

    kr_queue_release(gltf_object->buffers);
    kr_queue_release(gltf_object->buffer_views);
    kr_queue_release(gltf_object->accessors);
    kr_queue_release(gltf_object->meshes);
    kr_queue_release(gltf_object->nodes);
    kr_queue_release(gltf_object->images);
    kr_queue_release(gltf_object->textures);
    kr_queue_release(gltf_object->materials);
    kr_queue_release(gltf_object->samplers);
    kr_queue_release(gltf_object->cameras);

    return kr_success;
}

kr_error 
kr_scene_gltf_create(kr_scene* scene, const char* file_path, const char* base_path) {
    kr_error result = kr_success;

    char file_path_norm[256] = { 0 };
    kr_path_normalize(file_path, file_path_norm, sizeof(file_path_norm));

    char directory_path[256] = { 0 };
    kr_file_base(file_path_norm, directory_path, sizeof(directory_path));
    
    base_path = (base_path != kr_null) ? base_path : &directory_path[0];

	kr_size content_size = 0;
	result = kr_file_size(file_path_norm, &content_size);
    if (kr_success != result) return result;

	char* content_buffer = (char*)kr_allocate(content_size + 1);
	content_buffer[content_size] = 0;
	result = kr_file_contents(file_path_norm, content_buffer, content_size);

    kr_gltf_object gltf_object = { 0 };
    result = kr_geometry_gltf_parse(&gltf_object, content_buffer, directory_path);

    if (kr_success != result) {
        kr_free(&content_buffer);
        return result;
    }

    scene->aabb = kr_aabb_empty3();

    kr_queue_init(scene->instances, 1);
    kr_queue_init(scene->objects, 1);
    kr_queue_init(scene->materials, 1);
    kr_queue_init(scene->textures, 1);
    kr_queue_init(scene->samplers, 1);

    u32 face_count = 0;
    u32 attr_count = 0;

    kr_size mesh_index = 0;
    kr_size mesh_count = kr_queue_size(gltf_object.meshes);
    kr_object* meshes = kr_null;
    kr_object_instance* instances = kr_null;

    kr_queue_init(meshes, 1);
    kr_queue_init(instances, 1);

    for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
        kr_object   object = {0};
        kr_object_instance instance = {0};

		instance.model = kr_minvtransform4(kr_midentity4());

        kr_object_gltf_create(&object, kr_null, &gltf_object, mesh_index);

        face_count += (u32)object.as_mesh.face_count;
        attr_count += (u32)object.as_mesh.attr_count;

        scene->aabb = kr_aabb_expand(scene->aabb, object.aabb);

        instance.object_id = kr_queue_size(scene->objects);

        kr_queue_push(instances, instance);
        kr_queue_push(meshes, object);
    }

    kr_size node_index = 0;
    kr_size node_count = kr_queue_size(gltf_object.nodes);
    kr_size single_mesh_count = kr_queue_size(meshes);
    for (kr_size node_index = 0; node_index < node_count; node_index++) {
        kr_object   object = { 0 };
        kr_object_instance instance = { 0 };
        kr_gltf_node* gltf_node = &gltf_object.nodes[node_index];
        kr_object* gas_object = (gltf_node->mesh >= 0) ? &meshes[gltf_node->mesh] : kr_null;
       
        instance.model = kr_minvtransform4(kr_midentity4());

        if (!gas_object)
            continue;

        object.aabb = kr_aabb_empty3();

        object.type = gas_object->type;
        object.as_mesh.attr_count = gas_object->as_mesh.attr_count;
        object.as_mesh.face_count = gas_object->as_mesh.face_count;

        object.as_mesh.faces = kr_aligned_allocate(object.as_mesh.face_count * sizeof(*object.as_mesh.faces), kr_align_of(kr_vec3));
        object.as_mesh.vertices = kr_aligned_allocate(object.as_mesh.attr_count * sizeof(*object.as_mesh.vertices), kr_align_of(kr_vec3));
        object.as_mesh.normals = kr_aligned_allocate(object.as_mesh.attr_count * sizeof(*object.as_mesh.normals), kr_align_of(kr_vec3));
        object.as_mesh.uvs = kr_aligned_allocate(object.as_mesh.attr_count * sizeof(*object.as_mesh.uvs), kr_align_of(kr_vec3));

        memcpy(object.as_mesh.faces, gas_object->as_mesh.faces, object.as_mesh.face_count * sizeof(*object.as_mesh.faces));
        memcpy(object.as_mesh.vertices, gas_object->as_mesh.vertices, object.as_mesh.attr_count * sizeof(*object.as_mesh.vertices));
        memcpy(object.as_mesh.normals, gas_object->as_mesh.normals, object.as_mesh.attr_count * sizeof(*object.as_mesh.normals));
        memcpy(object.as_mesh.uvs, gas_object->as_mesh.uvs, object.as_mesh.attr_count * sizeof(*object.as_mesh.uvs));


        kr_gltf_node* parent = gltf_node;
        kr_mat4 view = gltf_node->transform.to;
        while (parent->parent >= 0) {
            parent = &gltf_object.nodes[parent->parent];
            view = kr_mmul4(parent->transform.to, view);
        } 
        kr_size attr_count = object.as_mesh.attr_count;
        for (kr_size attr_index = 0; attr_index < attr_count; attr_index++) {
            object.as_mesh.vertices[attr_index] = kr_vtransform3(view, object.as_mesh.vertices[attr_index]);
            object.as_mesh.normals[attr_index] = kr_ntransform3(view, object.as_mesh.normals[attr_index]);
        
            object.aabb = kr_aabb_expand3(object.aabb, object.as_mesh.vertices[attr_index]);
        }

        kr_log("Mesh Instance '%.*s' face count: %d attr_count: %d\n", gltf_node->name.length, gltf_node->name.view, object.as_mesh.face_count, object.as_mesh.attr_count);
        kr_log("Mesh Instance '%.*s' AABB [%f %f %f] x [%f %f %f]\n", gltf_node->name.length, gltf_node->name.view, object.aabb.min.x, object.aabb.min.y, object.aabb.min.z, object.aabb.max.x, object.aabb.max.y, object.aabb.max.z);

        kr_queue_push(scene->instances, instance);
        kr_queue_push(scene->objects, object);
    }

    kr_log("Scene face count: %d attr_count: %d\n", face_count, attr_count);
    kr_log("Scene AABB [%f %f %f] x [%f %f %f]\n", scene->aabb.min.x, scene->aabb.min.y, scene->aabb.min.z, scene->aabb.max.x, scene->aabb.max.y,  scene->aabb.max.z);
    
    kr_size texture_count = kr_queue_size(gltf_object.textures);
    for (kr_size texture_index = 0; texture_index < texture_count; texture_index++) {
        kr_texture texture = {0};
        char texture_uri[256] = {0};

        kr_gltf_texture* gltf_texture = &gltf_object.textures[texture_index];
        kr_gltf_image*  gltf_image = &gltf_object.images[gltf_texture->source];
        //kr_token_print(&gltf_image->uri);

        sprintf(texture_uri, "%s/%.*s", base_path, (i32)gltf_image->uri.length, gltf_image->uri.view);
        
        //result = kr_texture_create_from_file(&texture, texture_uri);
        
        kr_queue_push(scene->textures, texture);
    }

    kr_size sampler_count = kr_queue_size(gltf_object.samplers);
    for (kr_size sampler_index = 0; sampler_index < sampler_count; sampler_index++) {
        kr_sampler sampler = { 0 };
        kr_gltf_sampler* gltf_sampler = &gltf_object.samplers[sampler_index];

        sampler.wrapS = KR_WRAP_MODE_REPEAT;
        sampler.wrapT = KR_WRAP_MODE_REPEAT;

        kr_queue_push(scene->samplers, sampler);
    }

    kr_size material_count = kr_queue_size(gltf_object.materials);
    for (kr_size material_index = 0; material_index < material_count; material_index++) {
        kr_material material = {0};
        kr_gltf_material* gltf_material = &gltf_object.materials[material_index];
        kr_gltf_texture* gltf_texture = (gltf_material->base_color_texture.x < 0) ? kr_null : &gltf_object.textures[gltf_material->base_color_texture.x];

        material.roughness = gltf_material->roughness;
        material.metalicity = gltf_material->metalicity;
        material.base_color_texture = (gltf_material->base_color_texture.x < 0) ? kr_null : &scene->textures[gltf_material->base_color_texture.x];
        material.base_color_sampler = (kr_null == gltf_texture) ? kr_null : &scene->samplers[gltf_texture->sampler];
        material.base_color = gltf_material->base_color;

        kr_queue_push(scene->materials, material);
    }

    scene->instance_count = kr_queue_size(scene->instances);
    scene->object_count = kr_queue_size(scene->objects);


    // Camera
    kr_size camera_count = kr_queue_size(gltf_object.cameras);
    kr_size camera_index = 0;
    if(camera_count > 0) kr_queue_init(scene->cameras, 1);
    for (camera_index = 0; camera_index < camera_count; camera_index++ ) {
        kr_camera camera = { 0 };
        kr_gltf_camera* gltf_camera = &gltf_object.cameras[camera_index];
        if (gltf_camera->node >= 0) {
            kr_gltf_node* node = &gltf_object.nodes[gltf_camera->node];
            kr_gltf_node* parent = node;
            
            camera.type = gltf_camera->type;
            camera.view = node->transform;
            switch (camera.type) {
            case KR_CAMERA_PINHOLE:
                camera.as_pinhole.fov = gltf_camera->fov;
                camera.viewport.w = 10;
                camera.viewport.z = gltf_camera->ar * camera.viewport.w;
                break;
            default:
                break;
            }

            kr_queue_push(scene->cameras, camera);
        }
    }

    kr_gltf_object_destroy(&gltf_object);
    kr_free(&content_buffer);

    return kr_success;
}

kr_error
kr_scene_gltf_create_flat(kr_scene* scene, const char* file_path, const char* base_path) {
	kr_error result = kr_success;
	result = kr_scene_gltf_create(scene, file_path, base_path);
    if (result != kr_success) return result;

	u32 face_count = 0;
	u32 attr_count = 0;

	kr_size mesh_index = 0;
	kr_size mesh_count = kr_queue_size(scene->objects);

	kr_object uber_object = { 0 };
	for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		kr_object*   object = &scene->objects[mesh_index];

		face_count += (u32)object->as_mesh.face_count;
		attr_count += (u32)object->as_mesh.attr_count;
	}

	uber_object.type = KR_OBJECT_MESH;
    uber_object.aabb = kr_aabb_empty3();
	uber_object.as_mesh.face_count = face_count;
	uber_object.as_mesh.attr_count = attr_count;

    uber_object.as_mesh.faces = kr_aligned_allocate(uber_object.as_mesh.face_count * sizeof(*uber_object.as_mesh.faces), kr_align_of(kr_uvec4));
    uber_object.as_mesh.vertices = kr_aligned_allocate(uber_object.as_mesh.attr_count * sizeof(*uber_object.as_mesh.vertices), kr_align_of(kr_vec3));
    uber_object.as_mesh.normals = kr_aligned_allocate(uber_object.as_mesh.attr_count * sizeof(*uber_object.as_mesh.normals), kr_align_of(kr_vec3));
    uber_object.as_mesh.uvs = kr_aligned_allocate(uber_object.as_mesh.attr_count * sizeof(*uber_object.as_mesh.uvs), kr_align_of(kr_vec2));

	//kr_queue_reserve(uber_object.as_mesh.faces, uber_object.as_mesh.face_count);
	//kr_queue_reserve(uber_object.as_mesh.vertices, uber_object.as_mesh.attr_count);
	//kr_queue_reserve(uber_object.as_mesh.normals, uber_object.as_mesh.attr_count);
	//kr_queue_reserve(uber_object.as_mesh.uvs, uber_object.as_mesh.attr_count);

	u32 face_index = 0;
	u32 attr_offset = 0;
	u32 face_offset = 0;

	for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		kr_object*   object = &scene->objects[mesh_index];

		uvec4* faces = uber_object.as_mesh.faces + face_offset;
		vec3* vertices = uber_object.as_mesh.vertices + attr_offset;
		vec3* normals = uber_object.as_mesh.normals + attr_offset;
		vec2* uvs = uber_object.as_mesh.uvs + attr_offset;

		for (kr_size face_index = 0; face_index < object->as_mesh.face_count; face_index++) {
			uvec4 face = object->as_mesh.faces[face_index];
			faces[face_index] = (uvec4) {
				face.x + attr_offset,
					face.y + attr_offset,
					face.z + attr_offset,
					face.w
			};
		}

		kr_memcpy(vertices, object->as_mesh.vertices, object->as_mesh.attr_count * sizeof(*vertices));
		kr_memcpy(normals, object->as_mesh.normals, object->as_mesh.attr_count * sizeof(*normals));
		kr_memcpy(uvs, object->as_mesh.uvs, object->as_mesh.attr_count * sizeof(*uvs));

		uber_object.aabb = kr_aabb_expand(uber_object.aabb, object->aabb);

		face_offset += (u32)object->as_mesh.face_count;
		attr_offset += (u32)object->as_mesh.attr_count;
	}

    
	for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
		kr_object*   object = &scene->objects[mesh_index];

        kr_aligned_free(&object->as_mesh.faces);
        kr_aligned_free(&object->as_mesh.vertices);
        kr_aligned_free(&object->as_mesh.normals);
        kr_aligned_free(&object->as_mesh.uvs);
	}

	kr_queue_release(scene->objects);
	kr_queue_release(scene->instances);

	kr_queue_init(scene->instances, 1);
	kr_queue_init(scene->objects, 1);

	kr_object_instance instance = { 0 };

	instance.model = kr_minvtransform4(kr_midentity4());

	kr_queue_push(scene->instances, instance);
	kr_queue_push(scene->objects, uber_object);

	scene->instance_count = 1;
	scene->object_count = 1;

	return kr_success;
}

kr_error
kr_scene_gltf_create_from_bounds(kr_scene* scene, kr_aabb3* bounds, kr_size bounds_count) {
	kr_object* objects = kr_null;
	kr_object_instance* instances = kr_null;

	kr_queue_reserve(objects, bounds_count);
	kr_queue_reserve(instances, bounds_count);

	for (kr_size bounds_index = 0; bounds_index < bounds_count; bounds_index++) {
		kr_object* object = &objects[bounds_index];
		kr_object_instance* instance = &instances[bounds_index];

		object->type = KR_OBJECT_AABB;
		object->aabb = bounds[bounds_index];

		instance->object_id = (kr_index)bounds_index;
		instance->model = kr_minvtransform4(kr_midentity4());
	}

	scene->objects = objects;
	scene->instances = instances;

	scene->instance_count = kr_queue_size(scene->instances);
	scene->object_count = kr_queue_size(scene->objects);

	return kr_success;
}

kr_error
kr_scene_gltf_create_minecraft(kr_scene* scene) {

    u32 face_count = 0;
    u32 attr_count = 0;

    kr_size mesh_index = 0;
    kr_size mesh_count = kr_queue_size(scene->objects);

    for (mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
        kr_object* object = &scene->objects[mesh_index];

        face_count += (u32)object->as_mesh.face_count;
        attr_count += (u32)object->as_mesh.attr_count;
    }

    kr_object* minecraft_objects = kr_null;
    kr_object_instance* minecraft_instances = kr_null;
    kr_queue_reserve(minecraft_objects, face_count);
    kr_queue_reserve(minecraft_instances, face_count);

    kr_size aabb_index = 0;
    for (mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
        kr_object* object = &scene->objects[mesh_index];

        for (kr_size face_index = 0; face_index < object->as_mesh.face_count; face_index++) {
            kr_object_instance* aabb_instance = &minecraft_instances[aabb_index];
            kr_object* aabb_object = &minecraft_objects[aabb_index];
            uvec4 face = object->as_mesh.faces[face_index];

            vec3  va = object->as_mesh.vertices[face.x];
            vec3  vb = object->as_mesh.vertices[face.y];
            vec3  vc = object->as_mesh.vertices[face.z];

            aabb3 bbox = kr_aabb_empty3();

            bbox = kr_aabb_expand3(bbox, va);
            bbox = kr_aabb_expand3(bbox, vb);
            bbox = kr_aabb_expand3(bbox, vc);

            aabb_object->type = KR_OBJECT_AABB;
            aabb_object->aabb = bbox;

            aabb_instance->model = kr_minvtransform4(kr_midentity4());
            aabb_instance->object_id = aabb_index;

            aabb_index++;
        }
    }


    for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
        kr_object* object = &scene->objects[mesh_index];

        kr_queue_release(object->as_mesh.faces);
        kr_queue_release(object->as_mesh.vertices);
        kr_queue_release(object->as_mesh.normals);
        kr_queue_release(object->as_mesh.uvs);
    }

    kr_queue_release(scene->objects);
    kr_queue_release(scene->instances);

    scene->objects = minecraft_objects;
    scene->instances = minecraft_instances;

    scene->instance_count = kr_queue_size(scene->instances);
    scene->object_count = kr_queue_size(scene->objects);

    return kr_success;
}
