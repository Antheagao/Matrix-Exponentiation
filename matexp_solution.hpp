#ifndef MATEXP_SOLUTION_INCLUDED
#define MATEXP_SOLUTION_INCLUDED

#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include "archlab.h"
#include "tensor_t.hpp"
#include "function_map.hpp"

#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

/* Function to perform matrix multiplication at a target 8x speedup : O(n^3) */
template<typename T>
void __attribute__((noinline)) mult_solution(tensor_t<T> &C,
                                             const tensor_t<T> &A,
                                             const tensor_t<T> &B) {
	// Declare variables
	const uint32_t TILE_SIZE = 16;
	uint32_t cSizeX = C.size.x;
	uint32_t cSizeY = C.size.y;
	uint32_t bSizeX = B.size.x;
	uint32_t bSizeY = B.size.y;

	// Transpose the B matrix to improve cache performance : O(n^2)
	tensor_t<T> B_t(bSizeY, bSizeX);
	for (uint32_t index = 0; index < bSizeX * bSizeY; ++index) {
		uint32_t x = index / bSizeY;
		uint32_t y = index % bSizeY;
		B_t.get(y, x) = B.get(x, y);
	}

	// Initialize the C matrix to 0 : O(n^2)
	std::fill_n(C.data, cSizeX * cSizeY, 0);
	
    // Perform matrix multiplication using tiling : O(n^3)
	for(uint32_t i = 0; i < cSizeX; i += TILE_SIZE) {
		for(uint32_t k = 0; k < bSizeX; k += TILE_SIZE) {
			for(uint32_t j = 0; j < cSizeY; j += TILE_SIZE) {
				C.get(i, j) += A.get(i, k) * B.get(j, k);
			}
		}
	}
}

#pragma GCC pop_options

/* Function to create an identity matrix in the dst tensor argument */
template<typename T>
void __attribute__((noinline)) matexp_solution(tensor_t<T> &dst,
                                               const tensor_t<T> &A,
                                               uint32_t power) {
	// Declare variables
	uint32_t dstSizeX = dst.size.x;
	uint32_t dstSizeY = dst.size.y;
	uint32_t smallSize = std::min(dstSizeX, dstSizeY);

	// Set the diagonal elements to 1 to create an identity matrix: O(n^2)
	std::fill_n(dst.data, dstSizeX * dstSizeY, 0);
	for (uint32_t index = 0; index < smallSize; ++index) {
		dst.get(index, index) = 1;
	}

    // Perform matrix exponentiation using exponentiation by squaring: O(logn)
	tensor_t<T> A_pow(A);
	while (power > 0) {
		if (power & 1) {
			tensor_t<T> B(dst);
			mult_solution(dst, B, A_pow);
			--power;
		}
		else {
			tensor_t<T> B(A_pow);
			mult_solution(A_pow, B, B);
			power >>= 1;
		}
	}
}

#endif
