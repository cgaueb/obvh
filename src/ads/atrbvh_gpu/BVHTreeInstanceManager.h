#pragma once

#include "BVHTree.h"

/// <summary> Manager for BVH tree instances. Provides methods for allocating BVH structures in
///           host and device memory spaces, copying BVH structures between memory spaces and
///           freeing device space BVHs. Host space BVHs can be freed by calling their destructors.
/// </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
class BVHTreeInstanceManager
{
public:
    BVHTreeInstanceManager();
    ~BVHTreeInstanceManager();

    /// <summary> Allocates a BVH tree in host memory.
    ///
    ///           <para> This method does not construct BVH structures that are ready for ray
    ///                  tracing, it merely allocates them in memory. For full BVH construction, 
    ///                  check <see cref="BVHBuilder"/>. </para>
    /// </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="numberOfTriangles"> Number of triangles. </param>
    ///
    /// <returns> The allocated BVH tree, in host memory. </returns>
    BVHTree* CreateHostTree(unsigned int numberOfTriangles) const;

    /// <summary> Allocates a BVH tree in device memory.
    ///
    ///           <para> This method does not construct BVH structures that are ready for ray
    ///                  tracing, it merely allocates them in memory. For full BVH construction, 
    ///                  check <see cref="BVHBuilder"/>. </para>
    /// </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="numberOfTriangles"> Number of triangles. </param>
    ///
    /// <returns> The allocated BVH tree, in device memory. </returns>
    BVHTree* CreateDeviceTree(unsigned int numberOfTriangles) const;

    /// <summary> Copies a BVH structure from host to device memory. </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="hostTree"> The source tree in host memory. </param>
    ///
    /// <returns> A copy of the BVH, allocated in device memory. </returns>
    BVHTree* HostToDeviceTree(const BVHTree* hostTree) const;

    /// <summary> Copies a BVH structure from device to host memory. </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="deviceTree"> The source tree allocated in device memory. </param>
    ///
    /// <returns> A copy of the BVH, allocated in host memory. </returns>
    BVHTree* DeviceToHostTree(const BVHTree* deviceTree) const;

    /// <summary> Frees a BVH that was allocated in device memory.
    ///
    ///           <para> BVHs allocated in host memory can be freed by calling their destructors.
    ///                  </para> </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="deviceTree"> [out] The device memory BVH. </param>
    void FreeDeviceTree(BVHTree* deviceTree) const;

private:
    /// <summary> Creates a BVH in device memory. If a host memory tree is specified, copy its 
    ///           data to the newly created BVH.</summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="numberOfTriangles"> Number of triangles. </param>
    /// <param name="hostTree">          If non-null, the host tree from which to copy data from.
    ///                                  </param>
    ///
    /// <returns> The device memory BVH. </returns>
    BVHTree* CreateDeviceTree(unsigned int numberOfTriangles, const BVHTree* hostTree) const;

    /// <summary> Resets the temporary tree described by tempTree, setting all its pointers to
    ///           nullptr so data from the final tree is not wrongly freed. </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="tempTree"> The temporary tree. </param>
    void ResetTempTree(BVHTree* tempTree) const;
};
