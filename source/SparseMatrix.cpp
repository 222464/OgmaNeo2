// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseMatrix.h"

#include <stdio.h>

SparseMatrix::SparseMatrix() {}

SparseMatrix::SparseMatrix(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) {
	init(rows, columns, nonZeroValues, rowRanges, columnIndices);
}

SparseMatrix::SparseMatrix(
	const std::vector<float> &data,
	int rows,
	int columns
) {
	initFromMatrix(data, rows, columns);
}

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) {
	_rows = rows;
	_columns = columns;
	_nonZeroValues = nonZeroValues;
	_rowRanges = rowRanges;
	_columnIndices = columnIndices;
}

void SparseMatrix::initFromMatrix(
	const std::vector<float> &data,
	int rows,
	int columns
) {
	_rows = rows;
	_columns = columns;

	_rowRanges.reserve(rows);
	_rowRanges.push_back(0);

	int nonZeroCountInRow = 0; // Only need to set this to zero once because it's cumulative
	for (int row = 0; row < rows; ++row) {
		int row_offset = row * columns;

		for (int col = 0; col < columns; ++col) {
			int index = row_offset + col;

			if (data[index] != 0.0f) {
				_nonZeroValues.push_back(data[index]);
				_columnIndices.push_back(col);
				++nonZeroCountInRow;
			}
		}

		_rowRanges.push_back(nonZeroCountInRow);
	}
}

void SparseMatrix::initTranpose() {
	_columnRanges.resize(_columns + 1);
	_columnRanges[0] = 0;

	_rowIndices.resize(_nonZeroValues.size());

	// Temporary nonzero value count
	std::vector<int> nonZeroCountInColumns(_columns, 0);

	int nextIndex;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++) {
			_rowIndices[j] = i;

			nonZeroCountInColumns[_columnIndices[j]]++;
		}
	}

	// Build _columnRanges
	for (int i = 0; i < nonZeroCountInColumns.size(); i++)
		_columnRanges[i + 1] = _columnRanges[i] + nonZeroCountInColumns[i];
}

void SparseMatrix::multiplyVector(
	const std::vector<float> &in,
	std::vector<float> &out
) {
	int length = in.size();

	int nextIndex = 0;
	for (int i = 0; i < length; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; ++j) {
			out[i] += _nonZeroValues[j] * in[_columnIndices[j]];
		}
	}
}

void SparseMatrix::multiplyOneHotVectorArray(
	const OneHotVectorArray &arr,
	std::vector<float> &out                                                                                   
) {
	int length = out.size();

	int nextIndex = 0;
	for (int i = 0; i < length; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j += arr._vectorSize) {
			out[i] += _nonZeroValues[j + arr._indices[_columnIndices[j] / arr._vectorSize]];
		}
	}
}

void SparseMatrix::print(
	int elementWidth
) {
	std::vector<float> data;
	data.resize(_rows * _columns, 0.0f);

	int nextIndex = 0;
	for (int i = 0; i < _rows; i = nextIndex) {
		int rowOffset = i * _columns;
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; ++j) {
			float value = _nonZeroValues[j];
			int col = _columnIndices[j];

			data[rowOffset + col] = value;
		}
	}

	printf("[");

	for (int row = 0; row < _rows; ++row) {
		int rowOffset = row * _columns;

		if (row > 0) {
			printf(" [");
		}
		else {
			printf("[");
		}

		for (int col = 0; col < _columns; ++col) {
			float value = data[rowOffset + col];

			if (col < _columns - 1) {
				printf("%*.2f, ", elementWidth, value);
			}
			else {
				printf("%*.2f", elementWidth, value);
			}
		}

		printf("]");

		if (row < _rows - 1) {
			printf("\n");
		}
		else {
			printf("]\n");
		}
	}
}

void SparseMatrix::printT(
	int elementWidth
) {
	std::vector<float> data;
	data.resize(_columns * _rows, 0.0f);

	int nextIndex = 0;
	for (int i = 0; i < _columns; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; ++j) {
			float value = _nonZeroValues[j];
			int row = _rowIndices[j];

			data[_columnIndices[j] * _rows + row] = value;
		}
	}

	printf("[");

	for (int row = 0; row < _columns; ++row) {
		int rowOffset = row * _rows;

		if (row > 0) {
			printf(" [");
		}
		else {
			printf("[");
		}

		for (int col = 0; col < _rows; ++col) {
			float value = data[rowOffset + col];

			if (col < _rows - 1) {
				printf("%*.2f, ", elementWidth, value);
			}
			else {
				printf("%*.2f", elementWidth, value);
			}
		}

		printf("]");

		if (row < _columns - 1) {
			printf("\n");
		}
		else {
			printf("]\n");
		}
	}
}