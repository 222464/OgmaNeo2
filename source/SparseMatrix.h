// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "OneHotVectorArray.h"

// Compressed sparse row (CSR) format
struct SparseMatrix {
	int _rows;
	int _columns;

	std::vector<float> _nonZeroValues;

	std::vector<int> _rowRanges;
	std::vector<int> _columnIndices;

	std::vector<int> _columnRanges;
	std::vector<int> _rowIndices;

	SparseMatrix();

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

	void initFromMatrix(
		const std::vector<float> &data,
		int rows,
		int columns
	);

	// Needs to be called once before using transposed functions
	void initTranpose();

	// Size of "in" must equal size of "out"
	void multiplyVector(
		const std::vector<float> &in,
		std::vector<float> &out
	);

	void multiplyOneHotVectorArray(
		const OneHotVectorArray &arr,
		std::vector<float> &out
	);

	void print(int elementWidth);

	void printT(int elementWidth);
};