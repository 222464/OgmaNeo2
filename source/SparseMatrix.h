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

	// --- Init ---

	SparseMatrix() {}

	// If you don't want to construct immediately
	SparseMatrix(
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	) {
		create(nonZeroValues, rowRanges, columnIndices);
	}

	// From a non-compressed sparse matrix
	SparseMatrix(
		const std::vector<float> &data,
		int rows,
		int columns
	) {
		create(data, rows, columns);
	}

	// If you don't want to construct immediately
	void create(
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// From a non-compressed sparse matrix
	void create(
		const std::vector<float> &data,
		int rows,
		int columns
	);

	// --- Dense ---

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
};
} // namespace ogmaneo