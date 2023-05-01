#pragma once

#include "BVHTree.h"
#include "SceneWrapper.h"

/// <summary> BVHBuilder serves as a common interface for classes implementing different algorithms
///           for constructing BVH structures. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
class BVHBuilder
{
public:
    /// <summary> Builds a BVH tree. </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="sceneWrapper"> The scene wrapper. </param>
    ///
    /// <returns> The BVH tree. </returns>
    virtual BVHTree* BuildTree(const SceneWrapper* sceneWrapper, float ci, float ct, float* sah = nullptr, float* time_ms = nullptr) = 0;
};
