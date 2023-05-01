#include "korangar.h"
#include "queue.h"
#include "vecmath.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

kr_error kr_geometry_move(kr_object* object, vec3 move) {

	kr_size attr_count = kr_queue_size(object->as_mesh.vertices);
	for (kr_size attr_index = 0; attr_index < attr_count; attr_index++) {
		object->as_mesh.vertices[attr_index] = kr_vadd3(object->as_mesh.vertices[attr_index], move);
	}

    object->aabb.min = kr_vadd3(object->aabb.min, move);
    object->aabb.max = kr_vadd3(object->aabb.max, move);

	return kr_success;
}

kr_error kr_geometry_scale(kr_object* object, vec3 scale) {

	kr_size attr_count = kr_queue_size(object->as_mesh.vertices);
	for (kr_size attr_index = 0; attr_index < attr_count; attr_index++) {
		object->as_mesh.vertices[attr_index] = kr_vmul3(object->as_mesh.vertices[attr_index], scale);
	}

	object->aabb.min = kr_vmul3(object->aabb.min, scale);
	object->aabb.max = kr_vmul3(object->aabb.max, scale);

	return kr_success;
}

kr_error kr_geometry_scalef(kr_object* object, kr_scalar scale) {
	return kr_geometry_scale(object, (vec3) { scale, scale, scale });
}

kr_error kr_geometry_transform(kr_object* object, mat4 m) {

    object->aabb = kr_aabb_empty3();
	kr_size attr_count = object->as_mesh.attr_count;
	for (kr_size attr_index = 0; attr_index < attr_count; attr_index++) {
        object->as_mesh.vertices[attr_index] = kr_vtransform3(m, object->as_mesh.vertices[attr_index]);
        object->as_mesh.normals[attr_index]  = kr_ntransform3(m, object->as_mesh.normals[attr_index]);
   
        object->aabb = kr_aabb_expand3(object->aabb, object->as_mesh.vertices[attr_index]);
    }

	return kr_success;
}

kr_error kr_scene_transform(kr_scene* scene, mat4 m) {
    kr_size object_count = scene->object_count;
    kr_size instance_count = scene->instance_count;

    if (instance_count > 0) {
        for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
            kr_object_instance* instance = &scene->instances[instance_index];
            kr_object* object = &scene->objects[instance->object_id];

            switch (object->type) {
            case KR_OBJECT_AABB:

                break;
            case KR_OBJECT_MESH:
                kr_geometry_transform(object, m);
                break;
            default:
                break;
            }
        }
    }

    return kr_success;
}


kr_error kr_geometry_cube_create(kr_object* object) {

	const vec3 cube_vertex_data[] = {
	{ -0.500000, -0.500000, 0.500000 },  { 0.500000, -0.500000, 0.500000 },  { -0.500000, 0.500000, 0.500000 },
	{ -0.500000, 0.500000, 0.500000 },   { 0.500000, -0.500000, 0.500000 },  { 0.500000, 0.500000, 0.500000 },
	{ -0.500000, 0.500000, 0.500000 },   { 0.500000, 0.500000, 0.500000 },   { -0.500000, 0.500000, -0.500000 },
	{ -0.500000, 0.500000, -0.500000 },  { 0.500000, 0.500000, 0.500000 },   { 0.500000, 0.500000, -0.500000 },
	{ -0.500000, 0.500000, -0.500000 },  { 0.500000, 0.500000, -0.500000 },  { -0.500000, -0.500000, -0.500000 },
	{ -0.500000, -0.500000, -0.500000 }, { 0.500000, 0.500000, -0.500000 },  { 0.500000, -0.500000, -0.500000 },
	{ -0.500000, -0.500000, -0.500000 }, { 0.500000, -0.500000, -0.500000 }, { -0.500000, -0.500000, 0.500000 },
	{ -0.500000, -0.500000, 0.500000 },  { 0.500000, -0.500000, -0.500000 }, { 0.500000, -0.500000, 0.500000 },
	{ 0.500000, -0.500000, 0.500000 },   { 0.500000, -0.500000, -0.500000 }, { 0.500000, 0.500000, 0.500000 },
	{ 0.500000, 0.500000, 0.500000 },    { 0.500000, -0.500000, -0.500000 }, { 0.500000, 0.500000, -0.500000 },
	{ -0.500000, -0.500000, -0.500000 }, { -0.500000, -0.500000, 0.500000 }, { -0.500000, 0.500000, -0.500000 },
	{ -0.500000, 0.500000, -0.500000 },  { -0.500000, -0.500000, 0.500000 }, { -0.500000, 0.500000, 0.500000 },
	};

	const vec3 cube_normal_data[] = {
		{ 0.000000, 0.000000, 1.000000 },  { 0.000000, 0.000000, 1.000000 },  { 0.000000, 0.000000, 1.000000 },
		{ 0.000000, 0.000000, 1.000000 },  { 0.000000, 0.000000, 1.000000 },  { 0.000000, 0.000000, 1.000000 },
		{ 0.000000, 1.000000, 0.000000 },  { 0.000000, 1.000000, 0.000000 },  { 0.000000, 1.000000, 0.000000 },
		{ 0.000000, 1.000000, 0.000000 },  { 0.000000, 1.000000, 0.000000 },  { 0.000000, 1.000000, 0.000000 },
		{ 0.000000, 0.000000, -1.000000 }, { 0.000000, 0.000000, -1.000000 }, { 0.000000, 0.000000, -1.000000 },
		{ 0.000000, 0.000000, -1.000000 }, { 0.000000, 0.000000, -1.000000 }, { 0.000000, 0.000000, -1.000000 },
		{ 0.000000, -1.000000, 0.000000 }, { 0.000000, -1.000000, 0.000000 }, { 0.000000, -1.000000, 0.000000 },
		{ 0.000000, -1.000000, 0.000000 }, { 0.000000, -1.000000, 0.000000 }, { 0.000000, -1.000000, 0.000000 },
		{ 1.000000, 0.000000, 0.000000 },  { 1.000000, 0.000000, 0.000000 },  { 1.000000, 0.000000, 0.000000 },
		{ 1.000000, 0.000000, 0.000000 },  { 1.000000, 0.000000, 0.000000 },  { 1.000000, 0.000000, 0.000000 },
		{ -1.000000, 0.000000, 0.000000 }, { -1.000000, 0.000000, 0.000000 }, { -1.000000, 0.000000, 0.000000 },
		{ -1.000000, 0.000000, 0.000000 }, { -1.000000, 0.000000, 0.000000 }, { -1.000000, 0.000000, 0.000000 },
	};

	const vec2 cube_tex_coord_data[] = {
		{ 0.000000, 0.000000 }, { 1.000000, 0.000000 }, { 0.000000, 1.000000 }, { 0.000000, 1.000000 },
		{ 1.000000, 0.000000 }, { 1.000000, 1.000000 }, { 0.000000, 0.000000 }, { 1.000000, 0.000000 },
		{ 0.000000, 1.000000 }, { 0.000000, 1.000000 }, { 1.000000, 0.000000 }, { 1.000000, 1.000000 },
		{ 1.000000, 1.000000 }, { 0.000000, 1.000000 }, { 1.000000, 0.000000 }, { 1.000000, 0.000000 },
		{ 0.000000, 1.000000 }, { 0.000000, 0.000000 }, { 0.000000, 0.000000 }, { 1.000000, 0.000000 },
		{ 0.000000, 1.000000 }, { 0.000000, 1.000000 }, { 1.000000, 0.000000 }, { 1.000000, 1.000000 },
		{ 0.000000, 0.000000 }, { 1.000000, 0.000000 }, { 0.000000, 1.000000 }, { 0.000000, 1.000000 },
		{ 1.000000, 0.000000 }, { 1.000000, 1.000000 }, { 0.000000, 0.000000 }, { 1.000000, 0.000000 },
		{ 0.000000, 1.000000 }, { 0.000000, 1.000000 }, { 1.000000, 0.000000 }, { 1.000000, 1.000000 },
	};

	const uvec4 cube_face_data[] = {
		{ 0, 1, 2, 0 },    { 3, 4, 5, 0 },    { 6, 7, 8, 0 },    { 9, 10, 11, 0 },  { 12, 13, 14, 0 }, { 15, 16, 17, 0 },
		{ 18, 19, 20, 0 }, { 21, 22, 23, 0 }, { 24, 25, 26, 0 }, { 27, 28, 29, 0 }, { 30, 31, 32, 0 }, { 33, 34, 35, 0 },
	};

	object->type = KR_OBJECT_MESH;
	object->as_mesh.attr_count = 36;
	object->as_mesh.face_count = 12;

	kr_queue_reserve(object->as_mesh.faces, object->as_mesh.face_count);
	kr_queue_reserve(object->as_mesh.vertices, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.normals, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.uvs, object->as_mesh.attr_count);

	memcpy(object->as_mesh.vertices, cube_vertex_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.vertices));
	memcpy(object->as_mesh.normals, cube_normal_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.normals));
	memcpy(object->as_mesh.uvs, cube_tex_coord_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.uvs));

	memcpy(object->as_mesh.faces, cube_face_data, object->as_mesh.face_count * sizeof(*object->as_mesh.faces));

	object->aabb.max = (vec3){  0.5,  0.5,  0.5 };
	object->aabb.min = (vec3){ -0.5, -0.5, -0.5 };

	return kr_success;
}

kr_error kr_geometry_plane_create(kr_object* object) {
	const vec3 vertex_data[] = {
		{ -0.5, -0.5, 0 },  { 0.5, -0.5, 0 },  { 0.5, 0.5, 0 }, { -0.5, 0.5, 0 },
	};

	const vec3 normal_data[] = {
		{ 0, 0, -1 },  { 0, 0, -1 }, { 0, 0, -1 }, { 0, 0, -1 },
	};

	const vec2 tex_coord_data[] = {
		{ 0.000000, 0.000000 }, { 1.000000, 0.000000 }, { 0.000000, 1.000000 }, { 1.000000, 1.000000 }
	};

	const uvec4 face_data[] = {
		{ 0, 1, 2, 0 },
		{ 2, 3, 0, 0 },
	};

	object->type = KR_OBJECT_MESH;
	object->as_mesh.attr_count = 4;
	object->as_mesh.face_count = 2;

	kr_queue_reserve(object->as_mesh.faces, object->as_mesh.face_count);
	kr_queue_reserve(object->as_mesh.vertices, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.normals, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.uvs, object->as_mesh.attr_count);

	memcpy(object->as_mesh.vertices, vertex_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.vertices));
	memcpy(object->as_mesh.normals, normal_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.normals));
	memcpy(object->as_mesh.uvs, tex_coord_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.uvs));

	memcpy(object->as_mesh.faces, face_data, object->as_mesh.face_count * sizeof(*object->as_mesh.faces));

	object->aabb.max = (vec3){ 0.5,  0.5, 0 };
	object->aabb.min = (vec3){-0.5, -0.5, 0 };

	return kr_success;
}

kr_error kr_geometry_sphere_create(kr_object* object, u32 sector_count, u32 stack_count) {
    object->type = KR_OBJECT_MESH;
    
    float radius = 1.0f;
    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f;             // vertex normal
    float s, t;                                     // vertex texCoord

    sector_count = kr_clampi(sector_count, 8, sector_count);
    stack_count = kr_clampi(stack_count, 4, stack_count);
    float sectorStep = 2 * KR_PI / sector_count;
    float stackStep = KR_PI / stack_count;
    float sectorAngle, stackAngle;

    u32 count = 0;
    for (u32 i = 0; i <= stack_count; ++i)
    {
        // add (sectorCount+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for (u32 j = 0; j <= sector_count; ++j)
        {
            count++;
        }
    }

    object->as_mesh.attr_count = count;

    kr_queue_reserve(object->as_mesh.vertices, object->as_mesh.attr_count);
    kr_queue_reserve(object->as_mesh.normals, object->as_mesh.attr_count);
    kr_queue_reserve(object->as_mesh.uvs, object->as_mesh.attr_count);

    vec3 vertex_max = { 0, 0, 0 };
    vec3 vertex_min = { 0, 0, 0 };

    int attr_index = 0;
    for (u32 i = 0; i <= stack_count; ++i)
    {
        stackAngle = KR_PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for (u32 j = 0; j <= sector_count; ++j)
        {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            object->as_mesh.vertices[attr_index] = (vec3){ x, y, z };

            vertex_max = kr_vmax3(vertex_max, (vec3) { x, y, z });
            vertex_min = kr_vmin3(vertex_min, (vec3) { x, y, z });

            // normalized vertex normal (nx, ny, nz)
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;

            object->as_mesh.normals[attr_index] = (vec3){ nx, ny, nz };

            // vertex tex coord (s, t) range between [0, 1]
            s = (float)j / sector_count;
            t = (float)i / stack_count;

            object->as_mesh.uvs[attr_index] = (vec2){ s, t };

            attr_index++;
        }
    }

    int face_count = 0;

    for (u32 i = 0; i < stack_count; ++i)
    {
        for (u32 j = 0; j < sector_count; ++j)
        {
            if (i != 0)
            {
                face_count++;
            }

            // k1+1 => k2 => k2+1
            if (i != (stack_count - 1))
            {
                face_count++;
            }
        }
    }

    object->as_mesh.face_count = face_count;
    kr_queue_reserve(object->as_mesh.faces, object->as_mesh.face_count);

    int k1, k2;
    int face_index = 0;
    for (u32 i = 0; i < stack_count; ++i)
    {
        k1 = i * (sector_count + 1);     // beginning of current stack
        k2 = k1 + sector_count + 1;      // beginning of next stack

        for (u32 j = 0; j < sector_count; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1+1
            if (i != 0)
            {
                object->as_mesh.faces[face_index] = (kr_uvec4){ k1, k2, k1 + 1, 0 };
                face_index++;
            }

            // k1+1 => k2 => k2+1
            if (i != (stack_count - 1))
            {
                object->as_mesh.faces[face_index] = (kr_uvec4){ k1 + 1, k2, k2 + 1, 0 };
                face_index++;
            }
        }
    }

    object->aabb.max = vertex_max;
    object->aabb.min = vertex_min;

    FILE* obj = fopen("scene.obj", "w");
    FILE* mtl = fopen("scene.mtl", "w");

    int current_attr_count = 0;
    {
        fprintf(mtl, "newmtl material_%d\n", 0);
        fprintf(mtl, "\tNs %f\n", 44.0);
        fprintf(mtl, "\tNi %f\n", 1.5);
        fprintf(mtl, "\td %f\n", 1.0);
        fprintf(mtl, "\tTr %f\n", 0.0);
        fprintf(mtl, "\tTf %f %f %f\n", 1.0, 1.0, 1.0);
        fprintf(mtl, "\tillum %d\n", 2);
        float value = 0.8f;
        fprintf(mtl, "\tKa %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\tKd %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\tKs %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\tKe %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\n");
        fprintf(mtl, "\n");
    }
    fprintf(obj, "mtllib scene.mtl\n");
    {
        fprintf(obj, "o %s\n", "sphere");
        fprintf(obj, "g %s\n", "sphere");

        fprintf(obj, "\n");
        for (int attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
            fprintf(obj, "v %f %f %f\n", object->as_mesh.vertices[attr_index].x, object->as_mesh.vertices[attr_index].y, object->as_mesh.vertices[attr_index].z);
        }
        fprintf(obj, "\n");
        for (int attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
            fprintf(obj, "vt %f %f\n", object->as_mesh.uvs[attr_index].x, object->as_mesh.uvs[attr_index].y);
        }
        fprintf(obj, "\n");
        for (int attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
            fprintf(obj, "vn %f %f %f\n", object->as_mesh.normals[attr_index].x, object->as_mesh.normals[attr_index].y, object->as_mesh.normals[attr_index].z);
        }
        fprintf(obj, "\n");
        fprintf(obj, "usemtl material_%d\n", 0);
        for (int face_index = 0; face_index < object->as_mesh.face_count; face_index++) {
            uvec3 face = { object->as_mesh.faces[face_index].x + 1 + current_attr_count, object->as_mesh.faces[face_index].y + 1 + current_attr_count, object->as_mesh.faces[face_index].z + 1 + current_attr_count };
            fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
        }
        fprintf(obj, "\n");
        current_attr_count += (i32)object->as_mesh.attr_count;
    }
    fclose(obj);
    fclose(mtl);

    /*For tita As Integer = 0 To Density
        Dim vtita As Double = tita * (Math.PI / Density)
        For nphi As Integer = -Density To Density
        Dim vphi As Double = nphi * (Math.PI / Density)
        PointList(tita)(nphi + Density).X = Math.Sin(vtita) * Math.Cos(vphi)
        PointList(tita)(nphi + Density).Y = Math.Sin(vtita) * Math.Sin(vphi)
        PointList(tita)(nphi + Density).Z = Math.Cos(vtita)
        Next
        Next*/

	return kr_success;
}

kr_error kr_geometry_triangle_create(kr_object* object) {
	const vec3 vertex_data[] = {
		{ -1, 0, 0 },  { 1, 0, 0 },  { 0, 1, 0 },
	};

	const vec3 normal_data[] = {
		{ 0, 0, -1 },  { 0, 0, -1 },  { 0, 0, -1 },
	};

	const vec2 tex_coord_data[] = {
		{ 0.000000, 0.000000 }, { 1.000000, 0.000000 }, { 0.000000, 0.500000 }
	};

	const uvec4 face_data[] = {
		{ 0, 1, 2, 0 },  
	};

	object->type = KR_OBJECT_MESH;
	object->as_mesh.attr_count = 3;
	object->as_mesh.face_count = 1;

	kr_queue_reserve(object->as_mesh.faces, object->as_mesh.face_count);
	kr_queue_reserve(object->as_mesh.vertices, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.normals, object->as_mesh.attr_count);
	kr_queue_reserve(object->as_mesh.uvs, object->as_mesh.attr_count);

	memcpy(object->as_mesh.vertices, vertex_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.vertices));
	memcpy(object->as_mesh.normals, normal_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.normals));
	memcpy(object->as_mesh.uvs, tex_coord_data, object->as_mesh.attr_count * sizeof(*object->as_mesh.uvs));

	memcpy(object->as_mesh.faces, face_data, object->as_mesh.face_count * sizeof(*object->as_mesh.faces));

	object->aabb.max = (vec3){  1, 1, 0 };
	object->aabb.min = (vec3){ -1, 0, 0 };

	return kr_success;
}

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
	const char*  view;
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
            }

            unsigned long long exp_value = 0;
            while (1) {
                int digit = kr_char_to_digit(*token_at);
                if (digit < 0 || digit > 9) {
                    token_at--;
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

static kr_b32 kr_is_new_line(char character)
{
    return (character == '\n' || character == '\r');
}

static kr_error kr_token_next_string(kr_token* token)
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

kr_internal kr_size kr_array_size(kr_token token) {
    kr_size count = 0;
    while (KR_TOKEN_RBRACKET != token.type) {
        kr_token_next(&token);
        kr_token_next(&token);
        if (KR_TOKEN_COMMA != token.type && KR_TOKEN_RBRACKET != token.type) {
            return 0;
        }
        count++;
    }
    return count;
}

kr_internal kr_error kr_array_parse_vec2(kr_token* token, kr_vec2* out) {
    kr_uvec2 ret = { 0 };
    kr_token_next(token);

    out->x = kr_token_as_scalar(token);
    kr_token_next(token);
    if (token->type != KR_TOKEN_COMMA)
        return kr_success;
    kr_token_next(token);
    out->y = kr_token_as_scalar(token);

    return kr_success;
}

kr_internal kr_error kr_array_parse_vec3(kr_token* token, kr_vec3* out) {
    kr_uvec2 ret = { 0 };
    kr_token_next(token);

    out->x = kr_token_as_scalar(token);
    kr_token_next(token);
    if (token->type != KR_TOKEN_COMMA)
        return kr_success;
    kr_token_next(token);
    out->y = kr_token_as_scalar(token);
    kr_token_next(token);
    if (token->type != KR_TOKEN_COMMA)
        return kr_success;
    kr_token_next(token);
    out->z = kr_token_as_scalar(token);

    return kr_success;
}

kr_internal kr_error kr_property_parse(kr_token* token, kr_property* property) {
    if (token->type == KR_TOKEN_LBRACKET) {
        kr_size array_size = kr_array_size(*token);
        if (array_size == 2) {
            property->value.type = KR_PROPERTY_VEC2;
            kr_array_parse_vec2(token, &property->value.as_vec2);
        }
        else if (array_size == 3) {
            property->value.type = KR_PROPERTY_VEC3;
            kr_array_parse_vec3(token, &property->value.as_vec3);
        }
    }
    else {

    }
    return kr_success;
}

kr_error kr_scene_load_settings(kr_scene* scene, const char* filename) {
    kr_size content_size = 0;
    kr_file_size(filename, &content_size);
    char* content_buffer = (char*)kr_allocate(content_size + 1);
    content_buffer[content_size] = 0;
    kr_file_contents(filename, content_buffer, content_size);
    const char* data = content_buffer;

    kr_token token = { data, 0, KR_TOKEN_NONE };

    kr_property* properties = kr_null;
    kr_queue_init(properties, 1);

    while (token.type != KR_TOKEN_EOS)
    {
        if (token.type != KR_TOKEN_DPOUND)
        {
            kr_token_next(&token);
            continue;
        }
        kr_token section = { 0 };

        kr_token_next(&token);
        section = token;
        kr_token_next(&token);
        while (token.type != KR_TOKEN_DPOUND && token.type != KR_TOKEN_EOS)
        {
            if (token.type == KR_TOKEN_DPOUND) break;
            if (token.type != KR_TOKEN_LBRACE)
            {
                kr_token_next(&token);
                continue;
            }

            while (token.type != KR_TOKEN_RBRACE)
            {
                kr_token key = { 0 };
                kr_token value = { 0 };
                kr_token_next(&token);
                key = token;
                kr_token_next(&token);

                if (token.type != KR_TOKEN_COLON)
                    continue;

                kr_token_next(&token);
                kr_property property = { section, key, { 0 } };

                kr_property_parse(&token, &property);

                kr_queue_push(properties, property);
            }
            kr_token_next(&token);
        }
    }

    kr_size property_count = kr_queue_size(properties);
    for (kr_size property_index = 0; property_index < property_count; property_index++) {
        printf("Section: %.*s\n", (i32)properties[property_index].section.length, properties[property_index].section.view);
        printf("Key: %.*s\n", (i32)properties[property_index].key.length, properties[property_index].key.view);
        printf("\n");
    }

    return kr_success;
}

kr_error 
kr_scene_aabb_calculate(kr_scene* scene) {
    scene->aabb = kr_aabb_empty3();
	if (scene->instance_count) {
		for (kr_size instance_index = 0; instance_index < scene->instance_count; instance_index++) {
			kr_object_instance* instance = &scene->instances[instance_index];
			kr_object* object = &scene->objects[instance->object_id];

			switch (object->type) {
			case KR_OBJECT_SPHERE: {
				break;
			}
			case KR_OBJECT_AABB: {
				aabb3 aabb = kr_aabb_transform4(instance->model.from, object->aabb);
				scene->aabb = kr_aabb_expand(scene->aabb, aabb);
				break;
			}
			case KR_OBJECT_MESH: {
				aabb3 aabb = kr_aabb_transform4(instance->model.from, object->aabb);
				scene->aabb = kr_aabb_expand(scene->aabb, aabb);
				break;
			}
			default:
				//assert(0);
				break;
			}
		}
	} else {
		for (kr_size object_index = 0; object_index < scene->object_count; object_index++) {
			kr_object* object = &scene->objects[object_index];

			switch (object->type) {
			case KR_OBJECT_SPHERE: {
				break;
			}
			case KR_OBJECT_AABB: {
				scene->aabb = kr_aabb_expand(scene->aabb, object->aabb);
				break;
			}
			case KR_OBJECT_MESH: {
				scene->aabb = kr_aabb_expand(scene->aabb, object->aabb);
				break;
			}
			default:
				//assert(0);
				break;
			}
		}
	}


	return kr_success;
}

kr_error 
kr_scene_destroy(kr_scene* scene) {
    kr_size mesh_index = 0;
    kr_size mesh_count = scene->object_count;

    for (kr_size mesh_index = 0; mesh_index < mesh_count; mesh_index++) {
        kr_object* object = &scene->objects[mesh_index];

        switch (object->type) {
        case KR_OBJECT_SPHERE: {
            break;
        }
        case KR_OBJECT_AABB: {
            break;
        }
        case KR_OBJECT_MESH: {
            kr_aligned_free(&object->as_mesh.faces);
            kr_aligned_free(&object->as_mesh.vertices);
            kr_aligned_free(&object->as_mesh.normals);
            kr_aligned_free(&object->as_mesh.uvs);
        }
        default:
            break;
        }
    }

    kr_queue_release(scene->objects);
    kr_queue_release(scene->instances);
    kr_queue_release(scene->materials);
    kr_queue_release(scene->textures);

    return kr_success;
}

kr_error 
kr_scene_export_triangles_obj(kr_vec3* vertices, kr_size count, const char* filename) {
    char obj_name[256] = { 0 };
    char mtl_name[256] = { 0 };

    sprintf(obj_name, "%s.obj", filename);
    sprintf(mtl_name, "%s.mtl", filename);

    FILE* obj = fopen(obj_name, "w");
    FILE* mtl = fopen(mtl_name, "w");
    
    fprintf(mtl, "newmtl material_%d\n", 0);
    fprintf(mtl, "\tNs %f\n", 44.0);
    fprintf(mtl, "\tNi %f\n", 1.5);
    fprintf(mtl, "\td %f\n", 1.0);
    fprintf(mtl, "\tTr %f\n", 0.0);
    fprintf(mtl, "\tTf %f %f %f\n", 1.0, 1.0, 1.0);
    fprintf(mtl, "\tillum %d\n", 2);
    float value = 0.0f;
    fprintf(mtl, "\tKa %f %f %f %f\n", value, value, value, 1.0);
    fprintf(mtl, "\tKd %f %f %f %f\n", value, value, value, 1.0);
    fprintf(mtl, "\tKs %f %f %f %f\n", value, value, value, 1.0);
    fprintf(mtl, "\tKe %f %f %f %f\n", value, value, value, 1.0);
    fprintf(mtl, "\n");
    fprintf(mtl, "\n");

    fprintf(obj, "mtllib %\n", mtl_name);

    fprintf(obj, "o %s\n", "scene");
    fprintf(obj, "g %s\n", "scene");

    fprintf(obj, "\n");
    for (int attr_index = 0; attr_index < count; attr_index++) {
        fprintf(obj, "v %f %f %f\n", vertices[attr_index].x, vertices[attr_index].y, vertices[attr_index].z);
    }
    fprintf(obj, "\n");
    for (int attr_index = 0; attr_index < count; attr_index++) {
        fprintf(obj, "vt %f %f\n", 0.0f, 0.0f);
    }
    fprintf(obj, "\n");
    for (int attr_index = 0; attr_index < count; attr_index++) {
        fprintf(obj, "vn %f %f %f\n", 0.0f, 0.0f, 0.0f);
    }
    fprintf(obj, "\n");
    fprintf(obj, "usemtl material_%d\n", 0);
    for (int attr_index = 0; attr_index < count; attr_index += 3) {
        uvec3 face = { attr_index + 1, attr_index + 2, attr_index + 3 };
        fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
    }
    fprintf(obj, "\n");
    
    fclose(obj);
    fclose(mtl);

    return kr_success;
}

kr_error kr_scene_export_obj(kr_scene* scene, const char* filename) {
    char obj_name[256] = { 0 };
    char mtl_name[256] = { 0 };

    sprintf(obj_name, "%s.obj", filename);
    sprintf(mtl_name, "%s.mtl", filename);

    FILE* obj = fopen(obj_name, "w");
    FILE* mtl = fopen(mtl_name, "w");

    kr_size object_count = scene->object_count;
    kr_size instance_count = scene->instance_count;

    for (int i = 0; i < object_count; i++) {
        fprintf(mtl, "newmtl material_%d\n", i);
        fprintf(mtl, "\tNs %f\n", 44.0);
        fprintf(mtl, "\tNi %f\n", 1.5);
        fprintf(mtl, "\td %f\n", 1.0);
        fprintf(mtl, "\tTr %f\n", 0.0);
        fprintf(mtl, "\tTf %f %f %f\n", 1.0, 1.0, 1.0);
        fprintf(mtl, "\tillum %d\n", 2);
        float value = (float)(i + 1) / (float)(object_count);
        fprintf(mtl, "\tKa %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\tKd %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\tKs %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\tKe %f %f %f %f\n", value, value, value, 1.0);
        fprintf(mtl, "\n");
        fprintf(mtl, "\n");
    }

    for (kr_size instance_index = 0; instance_index < instance_count; instance_index++) {
        kr_object_instance* instance = &scene->instances[instance_index];
        kr_object* object = &scene->objects[instance->object_id];
        kr_object cleaned_object = { 0 };

        switch (object->type) {
        case KR_OBJECT_AABB:
            break;
        case KR_OBJECT_MESH: {


            int current_attr_count = 0;
            
            fprintf(obj, "mtllib %\n", mtl_name);
           
            fprintf(obj, "o %s\n", "scene");
            fprintf(obj, "g %s\n", "scene");

            fprintf(obj, "\n");
            for (int attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
                fprintf(obj, "v %f %f %f\n", object->as_mesh.vertices[attr_index].x, object->as_mesh.vertices[attr_index].y, object->as_mesh.vertices[attr_index].z);
            }
            fprintf(obj, "\n");
            for (int attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
                fprintf(obj, "vt %f %f\n", object->as_mesh.uvs[attr_index].x, object->as_mesh.uvs[attr_index].y);
            }
            fprintf(obj, "\n");
            for (int attr_index = 0; attr_index < object->as_mesh.attr_count; attr_index++) {
                fprintf(obj, "vn %f %f %f\n", object->as_mesh.normals[attr_index].x, object->as_mesh.normals[attr_index].y, object->as_mesh.normals[attr_index].z);
            }
            fprintf(obj, "\n");
            fprintf(obj, "usemtl material_%d\n", instance_index);
            for (int face_index = 0; face_index < object->as_mesh.face_count; face_index++) {
                uvec3 face = { object->as_mesh.faces[face_index].x + 1 + current_attr_count, object->as_mesh.faces[face_index].y + 1 + current_attr_count, object->as_mesh.faces[face_index].z + 1 + current_attr_count };
                fprintf(obj, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", face.x, face.x, face.x, face.y, face.y, face.y, face.z, face.z, face.z);
            }
            fprintf(obj, "\n");
            current_attr_count += object->as_mesh.attr_count;

        } break;
        default:
            break;
        }
    }

    fclose(obj);
    fclose(mtl);

    return kr_success;
}

kr_error kr_scene_load(kr_scene* scene, const char* filename) {
    

    return kr_success;
}
