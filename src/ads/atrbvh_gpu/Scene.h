#pragma once

#include <vector_types.h>

/// <summary> Scene representation. This structure will be used mostly to send scene data to GPU.
///           </summary>
///
/// <remarks> Leonardo, 12/16/2014. </remarks>
struct Scene
{
    unsigned int numberOfTriangles;
    float4* vertices;
    int4* indices;
    float3 boundingBoxMin;
    float3 boundingBoxMax;
};

struct BoxElements
{
    unsigned int numberOfElements;
    float4* centroids;
    float4* elem_boundingBoxMin;
    float4* elem_boundingBoxMax;
    float3 boundingBoxMin;
    float3 boundingBoxMax;
};
