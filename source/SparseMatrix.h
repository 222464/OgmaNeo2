#ifndef SPARSE_MATRIX_HEADER
#define SPARSE_MATRIX_HEADER

#include <vector>

#include "OneHotVectorArray.h"

// Compressed sparse row (CSR) format
struct SparseMatrix {
	int _rows;
	int _columns;
	std::vector<float> _nonZeroValues;
	std::vector<int> _rowRanges;
	std::vector<int> _columnIndices;

	SparseMatrix() {}

	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// From a non-compressed sparse matrix
	SparseMatrix(
		const std::vector<float> &data,
		int rows,
		int columns
	);

	// If you don't want to construct immediately
	void init(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// Size of "in" must equal size of "out"
	void multiplyVector(
		const std::vector<float> &in,
		std::vector<float> &out
	);

	void multiplyOneHotVectorArray(
		const OneHotVectorArray &arr,
		std::vector<float> &out
	);
};

#endif