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

	std::vector<int> _nonZeroValueIndices;
	std::vector<int> _columnRanges;
	std::vector<int> _rowIndices;

	// True if initTranspose was executed on this object
	bool _transposable;

	SparseMatrix();

	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices,
		bool generateTranspose = false
	);

	// From a non-compressed sparse matrix
	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &data,
		bool generateTranspose = false
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
		int rows,
		int columns,
		const std::vector<float> &data
	);

	// Needs to be called once before using transposed functions
	void initTranpose();

	// ---- All available operations below ----

	// Let A be *this* object ("this" pointer points to A)
	// Setting transposed to true will transpose A (*this* object)

	// -- AB or -(AB) (with negative = true) --
	void multiplyVector(
		const std::vector<float> &in,
		std::vector<float> &out,
		bool negative = false,
		bool transposed = false
	);

	void multiplyOHVA(
		const OneHotVectorArray &in,
		std::vector<float> &out,
		bool negative = false,
		bool transposed = false
	);

	// --- Output range versions ---

	// -- AB (with range) or -(AB) (with negative = true) --
	void multiplyRangeVector(
		const std::vector<float> &in,
		std::vector<float> &out,
		int startIndex,
		int length,
		bool negative = false,
		bool transposed = false
	);

	void multiplyRangeOHVA(
		const OneHotVectorArray &in,
		std::vector<float> &out,
		int startIndex,
		int length,
		bool negative = false,
		bool transposed = false
	);

	// ---- Print functions ----
	void print(int elementWidth);

	void printT(int elementWidth);
};