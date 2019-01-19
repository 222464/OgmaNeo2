// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

namespace ogmaneo {
// Compressed sparse row (CSR) format
struct SparseMatrix {
	std::vector<float> _nonZeroValues;
	std::vector<int> _rowRanges;
	std::vector<int> _columnIndices;

	// --- INIT ---

	SparseMatrix() {}

	SparseMatrix(
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
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// --- DENSE ---

	// Size of "in" must equal size of "out"
	void multiply(
		const std::vector<float> &in,
		std::vector<float> &out
	);

	// The range specifies which elements of "out" are to be computed
	void multiplyRange(
		const std::vector<float> &in,
		std::vector<float> &out,
		int startIndex,
		int length
	);

	// Treats "in" as having two dimensions
	void multiplyRectangularRange(
		const std::vector<float> &in,
		std::vector<float> &out,
		int startRow,
		int startColumn,
		int rectRows,
		int rectColumns,
		int rows,
		int columns
	);

	// --- OHERM ---

	// Multiply by a one-hot-row matrix
	void multiplyOHERM(
		const std::vector<int> &nonZeroIndices,
		int columns,
		std::vector<float> &out
	);

	// Multiply one row from a given one-hot-row matrix
	void multiplyOneRowOHERM(
		const std::vector<int> &nonZeroIndices,
		int row,
		int columns,
		std::vector<float> &out
	);

	// Multiply a range of rows from a given one-hot-row matrix
	void multiplyRangeOfRowOHERM(
		const std::vector<int> &nonZeroIndices,
		int startRow,
		int rowCount,
		int columns,
		std::vector<float> &out
	);
};
} // namespace ogmaneo