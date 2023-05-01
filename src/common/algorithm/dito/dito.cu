#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "common/korangar.h" 

#include "common/logger.h" 
#include "common/geometry.h"
#include "common/util.h"

#include "common/cuda/util.h"
#include "common/cuda/atomics.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "stdio.h"

#include "common/algorithm/dito/dito.cuh"

template <
    int THREAD_BLOCK_SIZE = 256
>
kr_inline_device
kr_scalar dito_block_min(kr_scalar* cache, kr_scalar value) {

    // Perform parallel reduction within the block.
    // TODO: Replace with shuffle down sync operations for the 32 block
    cache[threadIdx.x] = value;
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 1]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 2]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 4]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 8]);
    cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] = kr_min(cache[threadIdx.x], cache[threadIdx.x ^ 128]);

    return cache[threadIdx.x];
}


template <
    int THREAD_BLOCK_SIZE = 256
>
kr_inline_device
kr_scalar dito_block_max(kr_scalar* cache, kr_scalar value) {

    // Perform parallel reduction within the block.
    // TODO: Replace with shuffle down sync operations for the 32 block
    cache[threadIdx.x] = value;
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 1]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 2]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 4]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 8]);
    cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 16]);

    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 32]);

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 64]);

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] = kr_max(cache[threadIdx.x], cache[threadIdx.x ^ 128]);

    return cache[threadIdx.x];
}

template <
    int K = 7,
    int THREAD_BLOCK_SIZE = 256,
    int POINT_BLOCK_SIZE = 4
>
__global__ void
dito_minmax_proj(
    const kr_vec3* points, int N,
    kr_scalar* minProj, kr_scalar* maxProj
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    cvec3 point = (point_index < N) ? points[point_index] : points[0];
    kr_scalar proj;

    __shared__ kr_scalar proj_cache[THREAD_BLOCK_SIZE];

    // Slab 0: dir {1, 0, 0}
    proj = point.x;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[0], proj);
    }
    __syncthreads();
    proj = point.x;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[0], proj);
    }
    __syncthreads();
    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[1], proj);
    }
    __syncthreads();
    proj = point.y;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[1], proj);
    }
    __syncthreads();
    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[2], proj);
    }
    __syncthreads();
    proj = point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[2], proj);
    }
    __syncthreads();
    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[3], proj);
    }
    __syncthreads();
    proj = point.x + point.y + point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[3], proj);
    }
    __syncthreads();
    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[4], proj);
    }
    __syncthreads();
    proj = point.x + point.y - point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[4], proj);
    }
    __syncthreads();
    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[5], proj);
    }
    __syncthreads();
    proj = point.x - point.y + point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[5], proj);
    }
    __syncthreads();
    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    proj = dito_block_min<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMin(&minProj[6], proj);
    }
    __syncthreads();
    proj = point.x - point.y - point.z;
    proj = dito_block_max<THREAD_BLOCK_SIZE>(proj_cache, proj);
    if (threadIdx.x == 0) {
        atomicMax(&maxProj[6], proj);
    }
}


template <
    int K = 7,
    int THREAD_BLOCK_SIZE = 256,
    int POINT_BLOCK_SIZE = 4
>
__global__ void
dito_minmax_vert(
    cvec3* points, int N,
    const kr_scalar* minProj, const kr_scalar* maxProj,
    kr_i32* argMinVert, kr_i32* argMaxVert
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= N)
        return;

    cvec3 point = points[point_index];
    kr_scalar proj;

    // Slab 0: dir {1, 0, 0}
    proj = point.x;
    if (proj == minProj[0]) { atomicExch(&argMinVert[0], point_index); }
    if (proj == maxProj[0]) { atomicExch(&argMaxVert[0], point_index); }
    // Slab 1: dir {0, 1, 0}
    proj = point.y;
    if (proj == minProj[1]) { atomicExch(&argMinVert[1], point_index); }
    if (proj == maxProj[1]) { atomicExch(&argMaxVert[1], point_index); }
    // Slab 2: dir {0, 0, 1}
    proj = point.z;
    if (proj == minProj[2]) { atomicExch(&argMinVert[2], point_index); }
    if (proj == maxProj[2]) { atomicExch(&argMaxVert[2], point_index); }
    // Slab 3: dir {1, 1, 1}
    proj = point.x + point.y + point.z;
    if (proj == minProj[3]) { atomicExch(&argMinVert[3], point_index); }
    if (proj == maxProj[3]) { atomicExch(&argMaxVert[3], point_index); }
    // Slab 4: dir {1, 1, -1}
    proj = point.x + point.y - point.z;
    if (proj == minProj[4]) { atomicExch(&argMinVert[4], point_index); }
    if (proj == maxProj[4]) { atomicExch(&argMaxVert[4], point_index); }
    // Slab 5: dir {1, -1, 1}
    proj = point.x - point.y + point.z;
    if (proj == minProj[5]) { atomicExch(&argMinVert[5], point_index); }
    if (proj == maxProj[5]) { atomicExch(&argMaxVert[5], point_index); }
    // Slab 6: dir {1, -1, -1}
    proj = point.x - point.y - point.z;
    if (proj == minProj[6]) { atomicExch(&argMinVert[6], point_index); }
    if (proj == maxProj[6]) { atomicExch(&argMaxVert[6], point_index); }
}

kr_inline_host_device
kr_scalar getQualityValue(const vec3& len)
{
    return len.x * len.y + len.x * len.z + len.y * len.z; //half box area
//return len.x * len.y * len.z; //box volume
}

kr_inline_device
void findExtremalPoints_OneDir(vec3& normal, cvec3* vertArr, int nv,
    kr_scalar& minProj, kr_scalar& maxProj, vec3& minVert, vec3& maxVert)
{
    kr_scalar proj = kr_vdot3(vertArr[0], normal);

    // Declare som local variables to avoid aliasing problems
    kr_scalar tMinProj = proj, tMaxProj = proj;
    vec3 tMinVert = vertArr[0], tMaxVert = vertArr[0];

    for (int i = 1; i < nv; i++)
    {
        proj = kr_vdot3(vertArr[i], normal);
        if (proj < tMinProj) { tMinProj = proj; tMinVert = vertArr[i]; }
        if (proj > tMaxProj) { tMaxProj = proj; tMaxVert = vertArr[i]; }
    }

    // Transfer the result to the caller
    minProj = tMinProj;
    maxProj = tMaxProj;
    minVert = tMinVert;
    maxVert = tMaxVert;
}

kr_inline_device
void findExtremalProjs_OneDir(cvec3& normal, cvec3* vertArr, int nv, kr_scalar& minProj, kr_scalar& maxProj)
{
    kr_scalar proj = kr_vdot3(vertArr[0], normal);
    kr_scalar tMinProj = proj, tMaxProj = proj;

    for (int i = 1; i < nv; i++)
    {
        proj = kr_vdot3(vertArr[i], normal);
        tMinProj = min(tMinProj, proj);
        tMaxProj = max(tMaxProj, proj);
    }

    minProj = tMinProj;
    maxProj = tMaxProj;
}

kr_inline_device kr_scalar
findFurthestPointFromInfiniteEdge(vec3& p0, vec3& e0,
    cvec3* vertArr, int nv, vec3& p)
{
    kr_scalar sqDist, maxSqDist;
    int maxIndex = 0;

    maxSqDist = kr_vdistance_to_inf_edge3(vertArr[0], p0, e0);

    for (int i = 1; i < nv; i++)
    {
        sqDist = kr_vdistance_to_inf_edge3(vertArr[i], p0, e0);
        if (sqDist > maxSqDist)
        {
            maxSqDist = sqDist;
            maxIndex = i;
        }
    }
    p = vertArr[maxIndex];
    return maxSqDist;
}

kr_inline_device
void findFurthestPointPair(cvec3* minVert, cvec3* maxVert, int n,
    vec3& p0, vec3& p1)
{
    int indexFurthestPair = 0;
    kr_scalar sqDist, maxSqDist;
    maxSqDist = kr_vdistance3sqr(maxVert[0], minVert[0]);
    //printf("GPU SqDistance[%d] %f Min: {%f %f %f} Max: {%f %f %f}\n", 0, maxSqDist, minVert[0].x, minVert[0].y, minVert[0].z, maxVert[0].x, maxVert[0].y, maxVert[0].z);
    for (int k = 1; k < n; k++)
    {
        sqDist = kr_vdistance3sqr(maxVert[k], minVert[k]);
        //printf("GPU SqDistance[%d] %f Min: {%f %f %f} Max: {%f %f %f}\n", k, sqDist, minVert[k].x, minVert[k].y, minVert[k].z, maxVert[k].x, maxVert[k].y, maxVert[k].z);
        if (sqDist > maxSqDist) { maxSqDist = sqDist; indexFurthestPair = k; }
    }
    p0 = minVert[indexFurthestPair];
    p1 = maxVert[indexFurthestPair];
}

kr_inline_device
void findBestObbAxesFromTriangleNormalAndEdgeVectors(cvec3* vertArr, int nv, vec3& n,
    vec3& e0, vec3& e1, vec3& e2, vec3& b0, vec3& b1, vec3& b2, kr_scalar& bestVal)
{
    vec3 m0, m1, m2;
    vec3 dmax, dmin, dlen;
    kr_scalar quality;

    m0 = kr_vcross3(e0, n);
    m1 = kr_vcross3(e1, n);
    m2 = kr_vcross3(e2, n);

    // The operands are assumed to be orthogonal and unit normals	
    findExtremalProjs_OneDir(n, vertArr, nv, dmin.y, dmax.y);
    dlen.y = dmax.y - dmin.y;

    findExtremalProjs_OneDir(e0, vertArr, nv, dmin.x, dmax.x);
    findExtremalProjs_OneDir(m0, vertArr, nv, dmin.z, dmax.z);
    dlen.x = dmax.x - dmin.x;
    dlen.z = dmax.z - dmin.z;
    quality = getQualityValue(dlen);
    if (quality < bestVal) { bestVal = quality; b0 = e0; b1 = n; b2 = m0; }

    findExtremalProjs_OneDir(e1, vertArr, nv, dmin.x, dmax.x);
    findExtremalProjs_OneDir(m1, vertArr, nv, dmin.z, dmax.z);
    dlen.x = dmax.x - dmin.x;
    dlen.z = dmax.z - dmin.z;
    quality = getQualityValue(dlen);
    if (quality < bestVal) { bestVal = quality; b0 = e1; b1 = n; b2 = m1; }

    findExtremalProjs_OneDir(e2, vertArr, nv, dmin.x, dmax.x);
    findExtremalProjs_OneDir(m2, vertArr, nv, dmin.z, dmax.z);
    dlen.x = dmax.x - dmin.x;
    dlen.z = dmax.z - dmin.z;
    quality = getQualityValue(dlen);
    if (quality < bestVal) { bestVal = quality; b0 = e2; b1 = n; b2 = m2; }

}

kr_inline_device
void findUpperLowerTetraPoints(vec3& n, cvec3* selVertPtr, int np, vec3& p0,
    vec3& p1, vec3& p2, vec3& q0, vec3& q1, int& q0Valid, int& q1Valid)
{
    kr_scalar qMaxProj, qMinProj, triProj;
    kr_scalar eps = 0.000001f;

    q0Valid = q1Valid = 0;

    findExtremalPoints_OneDir(n, selVertPtr, np, qMinProj, qMaxProj, q1, q0);
    triProj = kr_vdot3(p0, n);

    if (qMaxProj - eps > triProj) { q0Valid = 1; }
    if (qMinProj + eps < triProj) { q1Valid = 1; }
}

kr_inline_device
int findBestObbAxesFromBaseTriangle(cvec3* minVert, cvec3* maxVert, int ns,
    cvec3* selVertPtr, int np, vec3& n, vec3& p0, vec3& p1, vec3& p2,
    vec3& e0, vec3& e1, vec3& e2, vec3& b0, vec3& b1, vec3& b2, kr_scalar& bestVal, kr_obb3& obb)
{
    kr_scalar sqDist;
    kr_scalar eps = 0.000001f;

    // Find the furthest point pair among the selected min and max point pairs
    findFurthestPointPair(minVert, maxVert, ns, p0, p1);

    // Degenerate case 1:
    // If the found furthest points are located very close, return OBB aligned with the initial AABB 
    if (kr_vdistance3sqr(p0, p1) < eps) { return 1; }

    // Compute edge vector of the line segment p0, p1 		
    e0 = kr_vnormalize3(kr_vsub3(p0, p1));

    // Find a third point furthest away from line given by p0, e0 to define the large base triangle
    sqDist = findFurthestPointFromInfiniteEdge(p0, e0, selVertPtr, np, p2);

    // Degenerate case 2:
    // If the third point is located very close to the line, return an OBB aligned with the line 
    if (sqDist < eps) { return 2; }

    // Compute the two remaining edge vectors and the normal vector of the base triangle				
    e1 = kr_vnormalize3(kr_vsub3(p1, p2));
    e2 = kr_vnormalize3(kr_vsub3(p2, p0));
    n = kr_vnormalize3(kr_vcross3(e1, e0));

    // Find best OBB axes based on the base triangle
    findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n, e0, e1, e2, b0, b1, b2, bestVal);

    return 0; // success
}

kr_inline_device
void findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(cvec3* selVertPtr, int np,
    vec3& n, vec3& p0, vec3& p1, vec3& p2, vec3& e0, vec3& e1,
    vec3& e2, vec3& b0, vec3& b1, vec3& b2, kr_scalar& bestVal, kr_obb3& obb)
{
    vec3 q0, q1;     // Top and bottom vertices for lower and upper tetra constructions
    vec3 f0, f1, f2; // Edge vectors towards q0; 
    vec3 g0, g1, g2; // Edge vectors towards q1; 
    vec3 n0, n1, n2; // Unit normals of top tetra tris
    vec3 m0, m1, m2; // Unit normals of bottom tetra tris		

    // Find furthest points above and below the plane of the base triangle for tetra constructions 
    // For each found valid point, search for the best OBB axes based on the 3 arising triangles
    int q0Valid, q1Valid;
    findUpperLowerTetraPoints(n, selVertPtr, np, p0, p1, p2, q0, q1, q0Valid, q1Valid);
    if (q0Valid)
    {
        f0 = kr_vnormalize3(kr_vsub3(q0, p0));
        f1 = kr_vnormalize3(kr_vsub3(q0, p1));
        f2 = kr_vnormalize3(kr_vsub3(q0, p2));
        n0 = kr_vnormalize3(kr_vcross3(f1, e0));
        n1 = kr_vnormalize3(kr_vcross3(f2, e1));
        n2 = kr_vnormalize3(kr_vcross3(f0, e2));
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n0, e0, f1, f0, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n1, e1, f2, f1, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n2, e2, f0, f2, b0, b1, b2, bestVal);
    }
    if (q1Valid)
    {
        g0 = kr_vnormalize3(kr_vsub3(q1, p0));
        g1 = kr_vnormalize3(kr_vsub3(q1, p1));
        g2 = kr_vnormalize3(kr_vsub3(q1, p2));
        m0 = kr_vnormalize3(kr_vcross3(g1, e0));
        m1 = kr_vnormalize3(kr_vcross3(g2, e1));
        m2 = kr_vnormalize3(kr_vcross3(g0, e2));
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m0, e0, g1, g0, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m1, e1, g2, g1, b0, b1, b2, bestVal);
        findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m2, e2, g0, g2, b0, b1, b2, bestVal);
    }
}

kr_inline_device kr_obb3 
kr_axis_aligned_obb(cvec3* mid, cvec3* len)
{
    kr_obb3 obb = { 0 };

    obb.mid = *mid;
    obb.ext = kr_vmul31(*len, 0.5f);
    obb.v0 = { 1, 0, 0 };
    obb.v1 = { 0, 1, 0 };
    obb.v2 = { 0, 0, 1 };

    return obb;
}

template <int K = 7>
__global__ void
dito_obb_candidate(
    const kr_vec3* points, int N,
    kr_scalar* minProj, kr_scalar* maxProj,
    kr_vec3* gminVert, kr_vec3* gmaxVert,
    kr_i32* argMinVert, kr_i32* argMaxVert,
    kr_obb3* obb
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= K)
        return;

    gminVert[point_index] = points[argMinVert[point_index]];
    gmaxVert[point_index] = points[argMaxVert[point_index]];

    if (point_index >= 1)
        return;

    vec3 minVert[K], maxVert[K];

    const kr_vec3* selVert = minVert;
    for (int i = 0; i < K; i++) {
        minVert[i] = points[argMinVert[i]];
        maxVert[i] = points[argMaxVert[i]];
    }

    int np = (N > 2 * K) ? 2 * K : N;
    const kr_vec3* selVertPtr = (N > 2 * K) ? minVert : points;

    vec3 p0, p1, p2; // Vertices of the large base triangle
    vec3 e0, e1, e2; // Edge vectors of the large base triangle
    vec3 n;
    vec3 bMin, bMax, bLen; // The dimensions of the oriented box
    kr_obb3 obb_candidate;

    vec3 alMid = { (minProj[0] + maxProj[0]) * 0.5f, (minProj[1] + maxProj[1]) * 0.5f, (minProj[2] + maxProj[2]) * 0.5f };
    vec3 alLen = { maxProj[0] - minProj[0], maxProj[1] - minProj[1], maxProj[2] - minProj[2] };
    kr_scalar alVal = getQualityValue(alLen);

    // Initialize the best found orientation so far to be the standard base
    kr_scalar bestVal = alVal;
    vec3 b0 = { 1, 0, 0 };
    vec3 b1 = { 0, 1, 0 };
    vec3 b2 = { 0, 0, 1 };

    int baseTriangleConstr = findBestObbAxesFromBaseTriangle(minVert, maxVert, K, selVertPtr, np, n, p0, p1, p2, e0, e1, e2, b0, b1, b2, bestVal, obb_candidate);
    /* Handle degenerate case */
    if (baseTriangleConstr != 0) { *obb = kr_axis_aligned_obb(&alMid, &alLen); return; }
#if 0
    if (point_index == 0) {

        printf("Base Trinagle %d %f:\n", baseTriangleConstr, bestVal);
        printf("Normal: {%f %f %f}\n", n.x, n.y, n.z);
        printf("P0	  : {%f %f %f}\n", p0.x, p0.y, p0.z);
        printf("P1    : {%f %f %f}\n", p1.x, p1.y, p1.z);
        printf("P2    : {%f %f %f}\n", p2.x, p2.y, p2.z);
        printf("B0	  : {%f %f %f}\n", b0.x, b0.y, b0.z);
        printf("B1    : {%f %f %f}\n", b1.x, b1.y, b1.z);
        printf("B2    : {%f %f %f}\n", b2.x, b2.y, b2.z);
        printf("E0	  : {%f %f %f}\n", e0.x, e0.y, e0.z);
        printf("E1    : {%f %f %f}\n", e1.x, e1.y, e1.z);
        printf("E2    : {%f %f %f}\n", e2.x, e2.y, e2.z);

        obb->v0 = b0;
        obb->v1 = b1;
        obb->v2 = b2;
        obb->ext = { 1, 1, 1 };
        obb->mid = { 0, 0, 0 };
    }
#endif
    // Find improved OBB axes based on constructed di-tetrahedral shape raised from base triangle
    findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(selVertPtr, np, n, p0, p1, p2, e0, e1, e2, b0, b1, b2, bestVal, obb_candidate);
#if 0
    if (point_index == 0) {

        printf("Tetras %d %f:\n", baseTriangleConstr, bestVal);
        printf("Normal: {%f %f %f}\n", n.x, n.y, n.z);
        printf("P0	  : {%f %f %f}\n", p0.x, p0.y, p0.z);
        printf("P1    : {%f %f %f}\n", p1.x, p1.y, p1.z);
        printf("P2    : {%f %f %f}\n", p2.x, p2.y, p2.z);
        printf("B0	  : {%f %f %f}\n", b0.x, b0.y, b0.z);
        printf("B1    : {%f %f %f}\n", b1.x, b1.y, b1.z);
        printf("B2    : {%f %f %f}\n", b2.x, b2.y, b2.z);
        printf("E0	  : {%f %f %f}\n", e0.x, e0.y, e0.z);
        printf("E1    : {%f %f %f}\n", e1.x, e1.y, e1.z);
        printf("E2    : {%f %f %f}\n", e2.x, e2.y, e2.z);

        obb->v0 = b0;
        obb->v1 = b1;
        obb->v2 = b2;
        obb->ext = { 1, 1, 1 };
        obb->mid = { 0, 0, 0 };
    }
#endif
    //computeObbDimensions(points, N, b0, b1, b2, bMin, bMax);

    obb->v0 = b0;
    obb->v1 = b1;
    obb->v2 = b2;
    obb->ext = { 0, 0, 0 };
    obb->mid = { 0, 0, 0 };

    bLen = kr_vsub3(bMax, bMin);
    bestVal = getQualityValue(bLen);
} 


template <int K = 7>
__global__ void
dito_obb_compute(
    const kr_vec3* points, int N,
    kr_scalar* minProj, kr_scalar* maxProj,
    kr_obb3* obb
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= N)
        return;

    vec3 bMin, bMax, bLen; // The dimensions of the oriented box

    cvec3 b0 = obb->v0;
    cvec3 b1 = obb->v1;
    cvec3 b2 = obb->v2;

    cvec3 point = points[point_index];

    kr_scalar proj;

    // We use some local variables to avoid aliasing problems
    kr_scalar* tMinProj = minProj;
    kr_scalar* tMaxProj = maxProj;

    proj = kr_vdot3(point, b0);
    atomicMin(tMinProj + 0, proj);
    atomicMax(tMaxProj + 0, proj);

    proj = kr_vdot3(point, b1);
    atomicMin(tMinProj + 1, proj);
    atomicMax(tMaxProj + 1, proj);

    proj = kr_vdot3(point, b2);
    atomicMin(tMinProj + 2, proj);
    atomicMax(tMaxProj + 2, proj);

}

kr_obb3
kr_cuda_points_obb(const kr_vec3* points, kr_size count, kr_scalar* p_elapsed_time) {
    cudaError_t cu_error;
    kr_scalar elapsed_ms = 0.0f;
    if (p_elapsed_time) *p_elapsed_time = 0;
    //kr_obb3 obb_cpu = kr_points_obb(points, count);
	kr_obb3 obb_gpu = { };

	printf("Point Count %d\n", count);

    constexpr auto KR_DITO_EXTERNAL_POINT_COUNT = 7;

    KR_ALLOC_DECLARE(kr_vec3, h_points, count);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_vec3, d_points, count);

    KR_ALLOC_DECLARE(kr_scalar, h_proj_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_scalar, d_proj_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);

    KR_ALLOC_DECLARE(kr_vec3, h_final_vert_minmax, 2);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_vec3, d_final_vert_minmax, 2);

    KR_ALLOC_DECLARE(kr_vec3, h_vert_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_vec3, d_vert_minmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);

    KR_ALLOC_DECLARE(kr_i32, h_vert_argminmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_i32, d_vert_argminmax, 2 * KR_DITO_EXTERNAL_POINT_COUNT);

    KR_ALLOC_DECLARE(kr_obb3, h_obb, sizeof(*h_obb));
    KR_CUDA_ALLOC_THRUST_DECLARE(kr_obb3, d_obb, sizeof(*h_obb));

    memcpy(h_points, points, count * sizeof(*h_points));
    cu_error = cudaMemcpy(thrust::raw_pointer_cast(d_points), h_points, count * sizeof(*h_points), cudaMemcpyHostToDevice);

    thrust::fill(d_vert_argminmax, d_vert_argminmax + 2 * KR_DITO_EXTERNAL_POINT_COUNT, -1);
    thrust::fill(d_final_vert_minmax + 0, d_final_vert_minmax + 1, kr_vec3{ FLT_MAX, FLT_MAX, FLT_MAX });
    thrust::fill(d_final_vert_minmax + 1, d_final_vert_minmax + 2, kr_vec3{ -FLT_MAX, -FLT_MAX, -FLT_MAX });
    thrust::fill(d_proj_minmax + 0, d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT, FLT_MAX);
    thrust::fill(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT, d_proj_minmax + 2 * KR_DITO_EXTERNAL_POINT_COUNT, -FLT_MAX);

    elapsed_ms = KernelLaunch().execute([&]() {
        constexpr auto KR_DITO_THREAD_BLOCK_SIZE = 256;
        constexpr auto KR_DITO_POINT_BLOCK_SIZE = 4;

        dim3 blockSize = dim3(KR_DITO_THREAD_BLOCK_SIZE);
        int bx = (count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        dito_minmax_proj
            <KR_DITO_EXTERNAL_POINT_COUNT, KR_DITO_THREAD_BLOCK_SIZE, KR_DITO_POINT_BLOCK_SIZE>
            << <gridSize, blockSize >> > (
                thrust::raw_pointer_cast(d_points), count,
                thrust::raw_pointer_cast(d_proj_minmax), thrust::raw_pointer_cast(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT)
                );
    });
    kr_log("'dito_minmax_proj' took %fms\n", elapsed_ms);
    if (p_elapsed_time) *p_elapsed_time += elapsed_ms;

    elapsed_ms = KernelLaunch().execute([&]() {
        constexpr auto KR_DITO_THREAD_BLOCK_SIZE = 256;
        constexpr auto KR_DITO_POINT_BLOCK_SIZE = 4;

        dim3 blockSize = dim3(KR_DITO_THREAD_BLOCK_SIZE);
        int bx = (count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        dito_minmax_vert
            <KR_DITO_EXTERNAL_POINT_COUNT, KR_DITO_THREAD_BLOCK_SIZE, KR_DITO_POINT_BLOCK_SIZE>
            << <gridSize, blockSize >> > (
                thrust::raw_pointer_cast(d_points), count,
                thrust::raw_pointer_cast(d_proj_minmax), thrust::raw_pointer_cast(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT),
                thrust::raw_pointer_cast(d_vert_argminmax), thrust::raw_pointer_cast(d_vert_argminmax + KR_DITO_EXTERNAL_POINT_COUNT)
                );
    });
    kr_log("'dito_minmax_vert' took %fms\n", elapsed_ms);
    if (p_elapsed_time) *p_elapsed_time += elapsed_ms;

    elapsed_ms = KernelLaunch().execute([&]() {
        dim3 blockSize = dim3(32);
        int bx = (KR_DITO_EXTERNAL_POINT_COUNT + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        dito_obb_candidate << <gridSize, blockSize >> > (
            thrust::raw_pointer_cast(d_points), count,
            thrust::raw_pointer_cast(d_proj_minmax), thrust::raw_pointer_cast(d_proj_minmax + KR_DITO_EXTERNAL_POINT_COUNT),
            thrust::raw_pointer_cast(d_vert_minmax), thrust::raw_pointer_cast(d_vert_minmax + KR_DITO_EXTERNAL_POINT_COUNT),
            thrust::raw_pointer_cast(d_vert_argminmax), thrust::raw_pointer_cast(d_vert_argminmax + KR_DITO_EXTERNAL_POINT_COUNT),
            thrust::raw_pointer_cast(d_obb)
        );
    });
    kr_log("'dito_obb_candidate' took %fms\n", elapsed_ms);
    if (p_elapsed_time) *p_elapsed_time += elapsed_ms;

    elapsed_ms = KernelLaunch().execute([&]() {
        constexpr auto KR_DITO_THREAD_BLOCK_SIZE = 256;
        constexpr auto KR_DITO_POINT_BLOCK_SIZE = 4;

        dim3 blockSize = dim3(KR_DITO_THREAD_BLOCK_SIZE);
        int bx = (count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);

        dito_obb_compute << <gridSize, blockSize >> > (
            thrust::raw_pointer_cast(d_points), count,
            (kr_scalar*)thrust::raw_pointer_cast(d_final_vert_minmax), (kr_scalar*)thrust::raw_pointer_cast(d_final_vert_minmax + 1),
            thrust::raw_pointer_cast(d_obb)
            );
    });
    kr_log("'dito_obb_compute' took %fms\n", elapsed_ms);
    if (p_elapsed_time) *p_elapsed_time += elapsed_ms;

    cu_error = cudaDeviceSynchronize();

    cu_error = cudaMemcpy(h_proj_minmax, thrust::raw_pointer_cast(d_proj_minmax), 2 * KR_DITO_EXTERNAL_POINT_COUNT * sizeof(*h_proj_minmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_vert_argminmax, thrust::raw_pointer_cast(d_vert_argminmax), 2 * KR_DITO_EXTERNAL_POINT_COUNT * sizeof(*h_vert_argminmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_vert_minmax, thrust::raw_pointer_cast(d_vert_minmax), 2 * KR_DITO_EXTERNAL_POINT_COUNT * sizeof(*h_vert_minmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_final_vert_minmax, thrust::raw_pointer_cast(d_final_vert_minmax), 2 * sizeof(*h_final_vert_minmax), cudaMemcpyDeviceToHost);
    cu_error = cudaMemcpy(h_obb, thrust::raw_pointer_cast(d_obb), sizeof(*h_obb), cudaMemcpyDeviceToHost);

    obb_gpu = *h_obb;

    cvec3 bMin = *(h_final_vert_minmax + 0);
    cvec3 bMax = *(h_final_vert_minmax + 1);
    cvec3 bLen = kr_vsub3(bMax, bMin);
    kr_scalar bestVal = getQualityValue(bLen);

    obb_gpu.ext = kr_vmul31(kr_vsub3(bMax, bMin), 0.5f);
    cvec3 mid_lcs = kr_vmul31(kr_vadd3(bMax, bMin), 0.5f);
    obb_gpu.mid = kr_vmul31(obb_gpu.v0, mid_lcs.x);
    obb_gpu.mid = kr_vadd3(obb_gpu.mid, kr_vmul31(obb_gpu.v1, mid_lcs.y));
    obb_gpu.mid = kr_vadd3(obb_gpu.mid, kr_vmul31(obb_gpu.v2, mid_lcs.z));

	return obb_gpu;
}
