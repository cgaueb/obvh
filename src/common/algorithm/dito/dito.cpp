/*
Supplemental C/C++ implementation of the DiTO OBB construction method described in the
chapter "Fast Computation of Tight-Fitting Oriented Bounding Boxes" of the book "Game 
Engine Gems 2".
*/

/*
This source file is distributed under the following BSD license:

Copyright 2011 Thomas Larsson and Linus Kallberg. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THOMAS LARSSON AND LINUS KALLBERG ``AS IS'' AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THOMAS LARSSON, LINUS 
KALLBERG, OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
Source file implementing the DiTO-14 algorithm.
Version: 1.0
*/
#define KR_VECMATH_IMPL
#include "common/vecmath.h"

#include "dito.h"
#include <cmath>
#include <cstdio>
#include <bitset>
#include <iostream>

namespace DiTO
{

/*
Explicit instantiations of the algorithm, to avoid having to put the function definition
in the header file. One of them can be commented out to save memory if it is not going to be used.
*/
template void DiTO_14<float>(Vector<float> vertArr[], int nv, OBB<float>& obb);
template void DiTO_14<double>(Vector<double> vertArr[], int nv, OBB<double>& obb);

template <typename F>
inline Vector<F> createVector(F x, F y, F z)
{	Vector<F> vector = { x, y, z };
	return vector;
}

template <typename F>
inline Vector<F> add(Vector<F> u, Vector<F> v)
{	Vector<F> sum = { u.x + v.x, u.y + v.y, u.z + v.z };
	return sum;
}

template <typename F>
inline Vector<F> sub(Vector<F> u, Vector<F> v)
{	Vector<F> diff = { u.x - v.x, u.y - v.y, u.z - v.z };
	return diff;
}

template <typename F>
inline Vector<F> scalVecProd(F s, Vector<F> u)
{	Vector<F> prod = { s * u.x, s * u.y, s * u.z };
	return prod;
}

template <typename F>
inline F dot(Vector<F> u, Vector<F> v) { return u.x*v.x + u.y*v.y + u.z*v.z; }

template <typename F>
inline Vector<F> cross(Vector<F> u, Vector<F> v)
{	Vector<F> prod = { u.y*v.z - u.z*v.y, -(u.x*v.z - u.z*v.x), u.x*v.y - u.y*v.x };
	return prod;
}

template <typename F>
inline F sqLength(Vector<F> u) { return u.x*u.x + u.y*u.y + u.z*u.z; }

template <typename F>
inline F length(Vector<F> u) { return (F) sqrt(sqLength(u)); }

template <typename F>
inline F sqPtPtDist(Vector<F> u, Vector<F> v) { return sqLength(sub(v, u)); }

template <typename F>
inline Vector<F> normalize(Vector<F> u)
{	F invLength = 1 / length(u);
	Vector<F> n = { u.x * invLength, u.y * invLength, u.z * invLength };
	return n;
}

template <typename F>
void findExtremalPoints_7FixedDirs(Vector<F> * vertArr, int nv, F minProj[7], F maxProj[7], 
	Vector<F> minVert[7], Vector<F> maxVert[7])
{	int i;
	F proj;

	// We use some local variables to avoid aliasing problems
	F tMinProj[7], tMaxProj[7];
	Vector<F> tMinVert[7], tMaxVert[7];
	
	// Slab 0: dir {1, 0, 0}
	proj = vertArr[0].x;
	tMinProj[0] = tMaxProj[0] = proj;
	tMinVert[0] = vertArr[0]; tMaxVert[0] = vertArr[0];
	// Slab 1: dir {0, 1, 0}
	proj = vertArr[0].y;
	tMinProj[1] = tMaxProj[1] = proj;
	tMinVert[1] = vertArr[0]; tMaxVert[1] = vertArr[0];
	// Slab 2: dir {0, 0, 1}
	proj = vertArr[0].z;
	tMinProj[2] = tMaxProj[2] = proj;
	tMinVert[2] = vertArr[0]; tMaxVert[2] = vertArr[0];
	// Slab 3: dir {1, 1, 1}
	proj = vertArr[0].x + vertArr[0].y + vertArr[0].z;
	tMinProj[3] = tMaxProj[3] = proj;
	tMinVert[3] = vertArr[0]; tMaxVert[3] = vertArr[0];
	// Slab 4: dir {1, 1, -1}
	proj = vertArr[0].x + vertArr[0].y - vertArr[0].z;
	tMinProj[4] = tMaxProj[4] = proj;
	tMinVert[4] = vertArr[0]; tMaxVert[4] = vertArr[0];
	// Slab 5: dir {1, -1, 1}
	proj = vertArr[0].x - vertArr[0].y + vertArr[0].z; 
	tMinProj[5] = tMaxProj[5] = proj;
	tMinVert[5] = vertArr[0]; tMaxVert[5] = vertArr[0];
	// Slab 6: dir {1, -1, -1}
	proj = vertArr[0].x - vertArr[0].y - vertArr[0].z;
	tMinProj[6] = tMaxProj[6] = proj;
	tMinVert[6] = vertArr[0]; tMaxVert[6] = vertArr[0];

	for (i = 1; i < nv; i++)
	{	// Slab 0: dir {1, 0, 0}
		proj = vertArr[i].x;
		if (proj < tMinProj[0]) { tMinProj[0] = proj; tMinVert[0] = vertArr[i]; }
		if (proj > tMaxProj[0]) { tMaxProj[0] = proj; tMaxVert[0] = vertArr[i]; }
		// Slab 1: dir {0, 1, 0}
		proj = vertArr[i].y;
		if (proj < tMinProj[1]) { tMinProj[1] = proj; tMinVert[1] = vertArr[i]; }
		if (proj > tMaxProj[1]) { tMaxProj[1] = proj; tMaxVert[1] = vertArr[i]; }
		// Slab 2: dir {0, 0, 1}
		proj = vertArr[i].z;
		if (proj < tMinProj[2]) { tMinProj[2] = proj; tMinVert[2] = vertArr[i]; }
		if (proj > tMaxProj[2]) { tMaxProj[2] = proj; tMaxVert[2] = vertArr[i]; }
		// Slab 3: dir {1, 1, 1}
		proj = vertArr[i].x + vertArr[i].y + vertArr[i].z;
		if (proj < tMinProj[3]) { tMinProj[3] = proj; tMinVert[3] = vertArr[i]; }
		if (proj > tMaxProj[3]) { tMaxProj[3] = proj; tMaxVert[3] = vertArr[i]; }
		// Slab 4: dir {1, 1, -1}
		proj = vertArr[i].x + vertArr[i].y - vertArr[i].z;
		if (proj < tMinProj[4]) { tMinProj[4] = proj; tMinVert[4] = vertArr[i]; }
		if (proj > tMaxProj[4]) { tMaxProj[4] = proj; tMaxVert[4] = vertArr[i]; }
		// Slab 5: dir {1, -1, 1}
		proj = vertArr[i].x - vertArr[i].y + vertArr[i].z; 		
		if (proj < tMinProj[5]) { tMinProj[5] = proj; tMinVert[5] = vertArr[i]; }
		if (proj > tMaxProj[5]) { tMaxProj[5] = proj; tMaxVert[5] = vertArr[i]; }
		// Slab 6: dir {1, -1, -1}
		proj = vertArr[i].x - vertArr[i].y - vertArr[i].z;
		if (proj < tMinProj[6]) { tMinProj[6] = proj; tMinVert[6] = vertArr[i]; }
		if (proj > tMaxProj[6]) { tMaxProj[6] = proj; tMaxVert[6] = vertArr[i]; }
	}

	// Note: Normalization of the extremal projection values can be done here. 
	// DiTO-14 only needs the extremal vertices, and the extremal projection values for slab 0-2 (to set the initial AABB).
	// Since unit normals are used for slab 0-2, no normalization is needed.
	// When needed, normalization of the remaining projection values can be done efficiently as follows:
	//tMinProj[3] *= 0.57735027f; tMaxProj[3] *= 0.57735027f;
	//tMinProj[4] *= 0.57735027f; tMaxProj[4] *= 0.57735027f;
	//tMinProj[5] *= 0.57735027f; tMaxProj[5] *= 0.57735027f;
	//tMinProj[6] *= 0.57735027f; tMaxProj[6] *= 0.57735027f;

	// Copy the result to the caller
	for (int i = 0; i < 7; ++i)
	{	minProj[i] = tMinProj[i];
		maxProj[i] = tMaxProj[i];
		minVert[i] = tMinVert[i];
		maxVert[i] = tMaxVert[i];
	}
}

template <typename F>
void findExtremalProjs_OneDir(Vector<F> & normal, Vector<F> * vertArr, int nv, F & minProj, F & maxProj) 
{	F proj = dot(vertArr[0], normal);
	F tMinProj = proj, tMaxProj = proj;
	
	for (int i = 1; i < nv; i++) 
	{	proj = dot(vertArr[i], normal);
		if (proj < tMinProj) { tMinProj = proj; }
		if (proj > tMaxProj) { tMaxProj = proj; }
	}

	minProj = tMinProj;
	maxProj = tMaxProj;
}

template <typename F>
void findExtremalPoints_OneDir(Vector<F> & normal, Vector<F> * vertArr, int nv,
	F & minProj, F & maxProj, Vector<F> & minVert, Vector<F> & maxVert)
{	F proj = dot(vertArr[0], normal);

	// Declare som local variables to avoid aliasing problems
	F tMinProj = proj, tMaxProj = proj;
	Vector<F> tMinVert = vertArr[0], tMaxVert = vertArr[0];
	
	for (int i = 1; i < nv; i++)
	{	proj = dot(vertArr[i], normal);
		if (proj < tMinProj) { tMinProj = proj; tMinVert = vertArr[i]; }
		if (proj > tMaxProj) { tMaxProj = proj; tMaxVert = vertArr[i]; }
	}

	// Transfer the result to the caller
	minProj = tMinProj;
	maxProj = tMaxProj;
	minVert = tMinVert;
	maxVert = tMaxVert;
}

template <typename F>
int sqDistPointInfiniteEdge(Vector<F> & q, Vector<F> & p0, Vector<F> & v, F & sqDist) 
{	Vector<F> u0 = sub(q, p0);
	F t = dot(v, u0);
	F sqLen_v = sqLength(v);
	sqDist = sqLength(u0) - t*t / sqLen_v;
	return 0;
}

template <typename F>
F findFurthestPointFromInfiniteEdge(Vector<F> & p0, Vector<F> & e0,
	Vector<F> * vertArr, int nv, Vector<F> & p)
{	F sqDist, maxSqDist; 
	int maxIndex = 0;
		
	sqDistPointInfiniteEdge(vertArr[0], p0, e0, maxSqDist);
	for (int i = 1; i < nv; i++)
	{	sqDistPointInfiniteEdge(vertArr[i], p0, e0, sqDist);

	if (sqDist > maxSqDist)
		{	maxSqDist = sqDist;
			maxIndex = i;
		}
	}

	p = vertArr[maxIndex];

	return maxSqDist;
}

template <typename F>
Vector<F> quantize(Vector<F>& v)
{
#ifdef KR_DITO_QUANTIZE
	return {
		(float)(int(v.x * 127.0f + (v.x > 0 ? 0.5f : -0.5f))) / 127.0f,
		(float)(int(v.y * 127.0f + (v.y > 0 ? 0.5f : -0.5f))) / 127.0f,
		(float)(int(v.z * 127.0f + (v.z > 0 ? 0.5f : -0.5f))) / 127.0f
	};
#else
	return v;
#endif
}

template <typename F>
F getQualityValue(Vector<F> & len)
{	return len.x * len.y + len.x * len.z + len.y * len.z; //half box area
	//return len.x * len.y * len.z; //box volume
}

template <typename F>
void findFurthestPointPair(Vector<F> * minVert, Vector<F> * maxVert, int n, 
	Vector<F> & p0, Vector<F> & p1)
{	int indexFurthestPair = 0;
	F sqDist, maxSqDist;
	maxSqDist = sqPtPtDist(maxVert[0], minVert[0]);
	for (int k = 1; k < n; k++)
	{	sqDist = sqPtPtDist(maxVert[k], minVert[k]);
		if (sqDist > maxSqDist) { maxSqDist = sqDist; indexFurthestPair = k; }
	}
	p0 = minVert[indexFurthestPair]; 
	p1 = maxVert[indexFurthestPair];
}

template <typename F>
void findUpperLowerTetraPoints(Vector<F> & n, Vector<F> * selVertPtr, int np, Vector<F> & p0, 
	Vector<F> & p1, Vector<F> & p2, Vector<F> & q0, Vector<F> & q1, int & q0Valid, int & q1Valid)
{	F qMaxProj, qMinProj, triProj;
	F eps = 0.000001f;

	q0Valid = q1Valid = 0;

	findExtremalPoints_OneDir(n, selVertPtr, np, qMinProj, qMaxProj, q1, q0);
	triProj = dot(p0, n);
	
	if (qMaxProj - eps > triProj) { q0Valid = 1; }
	if (qMinProj + eps < triProj) { q1Valid = 1; }
}

template <typename F>
void findBestObbAxesFromTriangleNormalAndEdgeVectors(Vector<F> * vertArr, int nv, Vector<F> & n, 
	Vector<F> & e0, Vector<F> & e1, Vector<F> & e2, Vector<F> & b0, Vector<F> & b1, Vector<F> & b2, F & bestVal)
{	Vector<F> m0, m1, m2;
	Vector<F> dmax, dmin, dlen;
	F quality;

	m0 = cross(e0, n); 
	m1 = cross(e1, n);	
	m2 = cross(e2, n);

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
	if (quality < bestVal) {  bestVal = quality; b0 = e1; b1 = n; b2 = m1; }

	findExtremalProjs_OneDir(e2, vertArr, nv, dmin.x, dmax.x);  
	findExtremalProjs_OneDir(m2, vertArr, nv, dmin.z, dmax.z);
	dlen.x = dmax.x - dmin.x;
	dlen.z = dmax.z - dmin.z;
	quality = getQualityValue(dlen);
	if (quality < bestVal) {  bestVal = quality; b0 = e2; b1 = n; b2 = m2; }
}

template <typename F>
int findBestObbAxesFromBaseTriangle(Vector<F> * minVert, Vector<F> * maxVert, int ns, 
	Vector<F> * selVertPtr, int np, Vector<F> & n, Vector<F> & p0, Vector<F> & p1, Vector<F> & p2, 
	Vector<F> & e0, Vector<F> & e1, Vector<F> & e2, Vector<F> & b0, Vector<F> & b1, Vector<F> & b2, F & bestVal, OBB<F> & obb)
{	F sqDist;
	F eps = 0.000001f;
	
	// Find the furthest point pair among the selected min and max point pairs
	findFurthestPointPair(minVert, maxVert, ns, p0, p1);

	// Degenerate case 1:
	// If the found furthest points are located very close, return OBB aligned with the initial AABB 
	if (sqPtPtDist(p0, p1) < eps) { return 1; }

	// Compute edge vector of the line segment p0, p1 		
	e0 = normalize(sub(p0, p1));

	// Find a third point furthest away from line given by p0, e0 to define the large base triangle
	sqDist = findFurthestPointFromInfiniteEdge(p0, e0, selVertPtr, np, p2); 

	// Degenerate case 2:
	// If the third point is located very close to the line, return an OBB aligned with the line 
	if (sqDist < eps) { return 2; }

	// Compute the two remaining edge vectors and the normal vector of the base triangle				
	e1 = normalize(sub(p1, p2));
	e2 = normalize(sub(p2, p0));
	n = normalize(cross(e1, e0));

	// Find best OBB axes based on the base triangle
	findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n, e0, e1, e2, b0, b1, b2, bestVal);

	return 0; // success
}

template <typename F>
void findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(Vector<F> * selVertPtr, int np, 
	Vector<F> & n, Vector<F> & p0, Vector<F> & p1, Vector<F> & p2, Vector<F> & e0, Vector<F> & e1, 
	Vector<F> & e2, Vector<F> & b0, Vector<F> & b1, Vector<F> & b2, F & bestVal, OBB<F> & obb)
{	Vector<F> q0, q1;     // Top and bottom vertices for lower and upper tetra constructions
	Vector<F> f0, f1, f2; // Edge vectors towards q0; 
	Vector<F> g0, g1, g2; // Edge vectors towards q1; 
	Vector<F> n0, n1, n2; // Unit normals of top tetra tris
	Vector<F> m0, m1, m2; // Unit normals of bottom tetra tris		
	
	// Find furthest points above and below the plane of the base triangle for tetra constructions 
	// For each found valid point, search for the best OBB axes based on the 3 arising triangles
	int q0Valid, q1Valid;
	findUpperLowerTetraPoints(n, selVertPtr, np, p0, p1, p2, q0, q1, q0Valid, q1Valid);
	if (q0Valid)
	{	f0 = normalize(sub(q0, p0));
		f1 = normalize(sub(q0, p1));
		f2 = normalize(sub(q0, p2));
		n0 = normalize(cross(f1, e0));
		n1 = normalize(cross(f2, e1));
		n2 = normalize(cross(f0, e2));
		findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n0, e0, f1, f0, b0, b1, b2, bestVal);
		findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n1, e1, f2, f1, b0, b1, b2, bestVal);
		findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, n2, e2, f0, f2, b0, b1, b2, bestVal);		
	}	
	if (q1Valid)
	{	g0 = normalize(sub(q1, p0));
		g1 = normalize(sub(q1, p1));
		g2 = normalize(sub(q1, p2));
		m0 = normalize(cross(g1, e0));
		m1 = normalize(cross(g2, e1));
		m2 = normalize(cross(g0, e2));
		findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m0, e0, g1, g0, b0, b1, b2, bestVal);			
		findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m1, e1, g2, g1, b0, b1, b2, bestVal);
		findBestObbAxesFromTriangleNormalAndEdgeVectors(selVertPtr, np, m2, e2, g0, g2, b0, b1, b2, bestVal);
	}
}

template <typename F>
void computeObbDimensions(Vector<F> * vertArr, int nv, Vector<F> & v0, Vector<F> & v1, Vector<F> & v2,
	Vector<F> & min, Vector<F> & max)
{	findExtremalProjs_OneDir(v0, vertArr, nv, min.x, max.x);  
	findExtremalProjs_OneDir(v1, vertArr, nv, min.y, max.y); 
	findExtremalProjs_OneDir(v2, vertArr, nv, min.z, max.z);
}

template <typename F>
void finalizeOBB(Vector<F> & v0, Vector<F> & v1, Vector<F> & v2, Vector<F> & min, Vector<F> & max, 
	Vector<F> & len, OBB<F> & obb)
{	obb.v0 = v0;
	obb.v1 = v1;
	obb.v2 = v2;
	obb.ext = scalVecProd<F>(0.5, len);
	Vector<F> q = scalVecProd<F>(0.5, add(min, max)); // q is the midpoint expressed in the OBB's own coordinate system 
	// Compute midpoint expressed in the standard base
	obb.mid = scalVecProd(q.x, v0);
	obb.mid = add(obb.mid, scalVecProd(q.y, v1));
	obb.mid = add(obb.mid, scalVecProd(q.z, v2));  
}

template <typename F>
void finalizeAxisAlignedOBB(Vector<F> & mid, Vector<F> & len, OBB<F> & obb)
{	obb.mid = mid;
	obb.ext = scalVecProd<F>(0.5, len);
	obb.v0 = createVector<F>(1, 0, 0);
	obb.v1 = createVector<F>(0, 1, 0);
	obb.v2 = createVector<F>(0, 0, 1);
}

template <typename F>
void finalizeLineAlignedOBB(Vector<F> & u, Vector<F> * vertArr, int nv, OBB<F> & obb)
{	// This function is only called if the construction of the large base triangle fails

	// Given u, build any orthonormal base u, v, w 
	Vector<F> r, v, w;
	
	// Make sure r is not equal to u
	r = u;
	if (fabs(u.x) > fabs(u.y) && fabs(u.x) > fabs(u.z)) { r.x = 0; }
	else if (fabs(u.y) > fabs(u.z) ) { r.y = 0; }
	else { r.z = 0; }
	
	F eps = 0.000001f;
	F sqLen = sqLength(r);
	if (sqLen < eps) { r.x = r.y = r.z = 1; }

	v = normalize(cross(u, r));
	w = normalize(cross(u, v));

	// compute the true obb dimensions by iterating over all vertices
	Vector<F> bMin, bMax, bLen;
	computeObbDimensions(vertArr, nv, u, v, w, bMin, bMax);
	bLen = sub(bMax, bMin);
	finalizeOBB(u, v, w, bMin, bMax, bLen, obb); // Assign all OBB params
}

template <typename F>
void DiTO_14(Vector<F> vertArr[], int nv, OBB<F>& obb)
{	const int ns = 7;		// Number of sample directions
	int np = ns * 2;		// Number of points selected along the sample directions
	Vector<F> selVert[ns * 2];	// Array holding selected points 	
	Vector<F> * selVertPtr;
	Vector<F> * minVert = selVert;			// Pointer to first half of selVert where the min points are placed
	Vector<F> * maxVert = selVert + ns;	// Pointer to the second half of selVert where the max points are placed		
	F minProj[ns];	// Array holding minimum projection values along the sample directions
	F maxProj[ns];	// Array holding maximum projection values along the sample directions	
	Vector<F> p0, p1, p2; // Vertices of the large base triangle
	Vector<F> e0, e1, e2; // Edge vectors of the large base triangle
	Vector<F> n;          // Unit normal of the large base triangle 
	Vector<F> alLen; // The axis-aligned dimensions of the vertices 
	Vector<F> alMid; // The axis-aligned mid point of the vertices 
	Vector<F> b0, b1, b2; // The currently best found OBB orientation  
	F alVal;	// Quality measure of the axis-aligned box 
	F bestVal;	// Estimate of box quality using axes b0, b1, b2
	Vector<F> bMin, bMax, bLen; // The dimensions of the oriented box

	if (nv <= 0)
	{	Vector<F> zero = createVector<F>(0, 0, 0);
		finalizeAxisAlignedOBB(zero, zero, obb);
		return;
	}

	// Select the extremal points along predefined slab directions
	findExtremalPoints_7FixedDirs(vertArr, nv, minProj, maxProj, minVert, maxVert);
	
	// Compute size of AABB (max and min projections of vertices are already computed as slabs 0-2)
	alMid = createVector((minProj[0] + maxProj[0]) * 0.5f, (minProj[1] + maxProj[1]) * 0.5f, (minProj[2] + maxProj[2]) * 0.5f);
	alLen = createVector(maxProj[0] - minProj[0], maxProj[1] - minProj[1], maxProj[2] - minProj[2]);
	alVal = getQualityValue(alLen);
	
	// Initialize the best found orientation so far to be the standard base
	bestVal = alVal; 
	b0 = createVector<F>(1, 0, 0);
	b1 = createVector<F>(0, 1, 0);
	b2 = createVector<F>(0, 0, 1);
	
	// Determine which points to use in the iterations below 
	if (nv > np) { selVertPtr = selVert; } // Use the selected extremal points 
	else { np = nv; selVertPtr = vertArr; } // Use all the input points since they are few

	// Find best OBB axes based on the constructed base triangle
	int baseTriangleConstr = findBestObbAxesFromBaseTriangle(minVert, maxVert, ns, selVertPtr, np, n, p0, p1, p2, e0, e1, e2, b0, b1, b2, bestVal, obb);
	if (baseTriangleConstr == 1) { finalizeAxisAlignedOBB(alMid, alLen, obb); return; } 
	if (baseTriangleConstr == 2) { finalizeLineAlignedOBB(e0, vertArr, nv, obb); return; }

	// Find improved OBB axes based on constructed di-tetrahedral shape raised from base triangle
	findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(selVertPtr, np, n, p0, p1, p2, e0, e1, e2, b0, b1, b2, bestVal, obb);

	// compute the true obb dimensions by iterating over all vertices
	computeObbDimensions(vertArr, nv, b0, b1, b2, bMin, bMax);
	bLen = sub(bMax, bMin);
	bestVal = getQualityValue(bLen);

	// Check if the OBB extent is still smaller than the intial AABB
	if (bestVal < alVal) { finalizeOBB(b0, b1, b2, bMin, bMax, bLen, obb); } // if so, assign all OBB params
	else { finalizeAxisAlignedOBB(alMid, alLen, obb); } // otherwise, assign all OBB params using the intial AABB
}

}