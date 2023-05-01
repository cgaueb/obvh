#pragma once

/// <summary> Sort the lists of keys and values using CUB's radix sort implementation
///           (http://nvlabs.github.io/cub/). Does not preserve input arrays. </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numberOfElements"> Number of elements. </param>
/// <param name="keysIn">           [in,out] Input keys. </param>
/// <param name="keysOut">          [in,out] Output keys. </param>
/// <param name="valuesIn">         [in,out] Input values. </param>
/// <param name="valuesOut">        [in,out] Output values. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceSort(unsigned int numberOfElements, unsigned int* keysIn, unsigned int* keysOut,
                 unsigned int* valuesIn, unsigned int* valuesOut);

/// <summary> Sort the lists of keys and values using CUB's radix sort implementation
///           (http://nvlabs.github.io/cub/). Does not preserve input arrays. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfElements"> Number of elements. </param>
/// <param name="keysIn">           [in,out] Input keys. </param>
/// <param name="keysOut">          [in,out] Output keys. </param>
/// <param name="valuesIn">         [in,out] Input values. </param>
/// <param name="valuesOut">        [in,out] Output values. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceSort(unsigned int numberOfElements, unsigned long long int* keysIn, 
        unsigned long long int* keysOut, unsigned int* valuesIn, unsigned int* valuesOut);