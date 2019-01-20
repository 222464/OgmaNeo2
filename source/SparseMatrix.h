// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include <assert.h>

namespace ogmaneo {
// Compressed sparse row (CSR) format
struct SparseMatrix {
	int _rows, _columns; // Dimensions

	std::vector<float> _nonZeroValues;
	std::vector<int> _rowRanges;
	std::vector<int> _columnIndices;

	// Transpose
	std::vector<int> _columnRanges;
	std::vector<int> _rowIndices;

	// --- Init ---

	SparseMatrix() {}

	// If you don't want to construct immediately
	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	) {
		create(rows, columns, nonZeroValues, rowRanges, columnIndices);
	}

	// From a non-compressed sparse matrix
	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &data
	) {
		create(rows, columns, data);
	}

	// If you don't want to construct immediately
	void create(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// From a non-compressed sparse matrix
	void create(
		int rows,
		int columns,
		const std::vector<float> &data
	);

	// Generate a transpose, must be called after the original has been created
	void createT();

	// --- Dense ---

	void multiply(
		const std::vector<float> &in,
		std::vector<float> &out
	);

	// The range specifies which elements of "out" are to be computed
	void multiplyRange(
		const std::vector<float> &in,
		std::vector<float> &out,
		int startRow,
		int rowCount
	);

	// --- Transpose ---

	void multiplyT(
		const std::vector<float> &in,
		std::vector<float> &out
	);

	// The range specifies which elements of "out" are to be computed
	void multiplyRangeT(
		const std::vector<float> &in,
		std::vector<float> &out,
		int startRow,
		int rowCount
	);

	// --- One-Hot Vectors Operations ---

	// Multiply by a one-hot-row matrix
	void multiplyOHVs(
		const std::vector<int> &nonZeroIndices,
		std::vector<float> &out,
		int oneHotSize,
		bool negative = false
	);

	// Multiply a range of rows from a given one-hot-row matrix
	void multiplyRangeOfRowOHVs(
		const std::vector<int> &nonZeroIndices,
		std::vector<float> &out,
		int startRow,
		int rowCount,
		int oneHotSize,
		bool negative = false
	);

	// --- One-Hot Vectors Operations: Transpose ---

	// Multiply by a one-hot-row matrix
	void multiplyOHVsT(
		const std::vector<int> &nonZeroIndices,
		std::vector<float> &out,
		int oneHotSize,
		bool negative = false
	);

	// Multiply a range of rows from a given one-hot-row matrix
	void multiplyRangeOfRowOHVsT(
		const std::vector<int> &nonZeroIndices,
		std::vector<float> &out,
		int startRow,
		int rowCount,
		int oneHotSize,
		bool negative = false
	);

	// --- Matrix Modification Rules ---

	// For dense deltas
	void deltaRuleRangeOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &deltas,
		int startRow,
		int rowCount,
		int oneHotSize
	);

	// For when deltas are also OHVs
	void deltaOHVRuleOHVs(
		const std::vector<int> &nonZeroIndices,
		int OHVIndex,
		int inputOneHotSize,
		int outputOneHotSize,
		int positiveIndex,
		int negativeIndex,
		float alpha
	);

	void deltaOHVRuleOHVsT(
		const std::vector<int> &nonZeroIndices,
		int OHVIndex,
		int inputOneHotSize,
		int outputOneHotSize,
		int positiveIndex,
		int negativeIndex,
		float alpha
	);
};
} // namespace ogmaneo