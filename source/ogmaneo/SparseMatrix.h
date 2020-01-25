// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <math.h>
#include <assert.h>

namespace ogmaneo {
// Compressed sparse row (CSR) format
struct SparseMatrix {
	int rows, columns; // Dimensions

	std::vector<float> nonZeroValues;
	std::vector<int> rowRanges;
	std::vector<int> columnIndices;

	// Transpose
	std::vector<int> nonZeroValueIndices;
	std::vector<int> columnRanges;
	std::vector<int> rowIndices;

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
		init(rows, columns, nonZeroValues, rowRanges, columnIndices);
	}

	// From a non-compressed sparse matrix
	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &data
	) {
		init(rows, columns, data);
	}

	// If you don't want to construct immediately
	void init(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// From a non-compressed sparse matrix
	void init(
		int rows,
		int columns,
		const std::vector<float> &data
	);

	// Generate a transpose, must be called after the original has been created
	void initT();

	float multiply(
		const std::vector<float> &in,
		int row
	);

	float multiplyT(
		const std::vector<float> &in,
		int column
	);

	float multiplyNoDiagonal(
		const std::vector<float> &in,
		int row
	);

	float multiplyNoDiagonalT(
		const std::vector<float> &in,
		int column
	);

	int count(
		int row
	);

	int countT(
		int column
	);

	float total(
		const std::vector<float> &in,
		int row
	);

	float totalT(
		const std::vector<float> &in,
		int column
	);

	void deltas(
		const std::vector<float> &in,
		float delta,
		int row
	);

	void deltasT(
		const std::vector<float> &in,
		float delta,
		int column
	);

	void normalize(
		int row
	);

	void normalizeT(
		int column
	);

	float magnitude2(
		int row
	);

	float magnitude2T(
		int column
	);

	void copyRow(
		const SparseMatrix &source,
		int row
	);

	void copyColumn(
		const SparseMatrix &source,
		int column
	);

	void hebb(
		const std::vector<float> &in,
		int row,
		float post,
		float alpha
	);

	void hebbT(
		const std::vector<float> &in,
		int column,
		float post,
		float alpha
	);
};
} // namespace ogmaneo