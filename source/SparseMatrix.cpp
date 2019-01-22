// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseMatrix.h"

#include <stdio.h>

#include <assert.h>

SparseMatrix::SparseMatrix() {}

SparseMatrix::SparseMatrix(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices,
	bool generateTranspose
) {
	init(rows, columns, nonZeroValues, rowRanges, columnIndices);

	if (generateTranspose) initTranpose();
}

SparseMatrix::SparseMatrix(
	int rows,
	int columns,
	const std::vector<float> &data,
	bool generateTranspose
) {
	initFromMatrix(rows, columns, data);

	if (generateTranspose) initTranpose();
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
	int rows,
	int columns,
	const std::vector<float> &data
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
	_nonZeroValueIndices.resize(_nonZeroValues.size());

	_columnRanges.resize(_columns + 1, 0);

	_rowIndices.resize(_nonZeroValues.size());

	int nextIndex;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++) {
			_columnRanges[_columnIndices[j]]++;
		}
	}

	// This loop fixes column ranges.
	// Before this loop column ranges
	// just has the number of non-zeroes
	// in each column, rather than the ranges
	// respresented between two values in column ranges
	int offset = 0;

	for (int i = 0; i < _columns; ++i) {
		int temp = _columnRanges[i];

		_columnRanges[i] = offset;

		offset += temp;
	}

	_columnRanges[_columns] = offset;

	// Generate row indices and non-zero value indices
	std::vector<int> columnOffsets = _columnRanges;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; ++j) {
			int columnIndex = _columnIndices[j];

			int nonZeroValueIndexIndex = columnOffsets[columnIndex];

			_rowIndices[nonZeroValueIndexIndex] = i;

			_nonZeroValueIndices[nonZeroValueIndexIndex] = j;

			++columnOffsets[columnIndex];
		}
	}
}

void SparseMatrix::multiplyVector(
	const std::vector<float> &in,
	std::vector<float> &out,
	bool negative,
	bool transposed
) {
	if (transposed)
		multiplyRangeVector(in, out, 0, _columns, negative, transposed);
	else
		multiplyRangeVector(in, out, 0, _rows, negative, transposed);
}

void SparseMatrix::multiplyOHVA(
	const OneHotVectorArray &in,
	std::vector<float> &out,
	bool negative,
	bool transposed
) {
	if (transposed)
		multiplyRangeOHVA(in, out, 0, _columns, negative, transposed);
	else
		multiplyRangeOHVA(in, out, 0, _rows, negative, transposed);
}

void SparseMatrix::multiplyRangeVector(
	const std::vector<float> &in,
	std::vector<float> &out,
	int startIndex,
	int length,
	bool negative,
	bool transposed
) {
	int end = startIndex + length;

	if (transposed) {
		assert(_nonZeroValueIndices.size() > 0);

		if (negative) {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; j++) {
					out[i] -= _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];
				}
			}
		}
		else {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; j++) {
					out[i] += _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];
				}
			}
		}
	}
	else {
		if (negative) {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; ++j) {
					out[i] -= _nonZeroValues[j] * in[_columnIndices[j]];
				}
			}
		}
		else {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; ++j) {
					out[i] += _nonZeroValues[j] * in[_columnIndices[j]];
				}
			}
		}
	}
}

void SparseMatrix::multiplyRangeOHVA(
	const OneHotVectorArray &arr,
	std::vector<float> &out,
	int startIndex,
	int length,
	bool negative,
	bool transposed
) {
	int end = startIndex + length;

	if (transposed) {
		assert(_nonZeroValueIndices.size() > 0);

		if (negative) {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; j += arr._vectorSize) {
					out[i] -= _nonZeroValues[_nonZeroValueIndices[_rowIndices[j] / arr._vectorSize] + j];
				}
			}
		}
		else {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; j += arr._vectorSize) {
					out[i] += _nonZeroValues[_nonZeroValueIndices[_rowIndices[j] / arr._vectorSize] + j];
				}
			}
		}
	}
	else {
		if (negative) {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j += arr._vectorSize) {
					out[i] -= _nonZeroValues[_nonZeroValueIndices[_columnIndices[j] / arr._vectorSize] + j];
				}
			}
		}
		else {
			int nextIndex;

			for (int i = startIndex; i < end; i = nextIndex) {
				nextIndex = i + 1;

				for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j += arr._vectorSize) {
					out[i] += _nonZeroValues[_nonZeroValueIndices[_columnIndices[j] / arr._vectorSize] + j];
				}
			}
		}
	}
}

void SparseMatrix::print(
	int elementWidth,
	int precision,
	bool transposed
) {
	if (transposed) {
		assert(_nonZeroValueIndices.size() > 0);

		std::vector<float> data;
		data.resize(_columns * _rows, 0.0f);

		int nextIndex = 0;
		for (int i = 0; i < _columns; i = nextIndex) {
			nextIndex = i + 1;

			for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; ++j) {
				float value = _nonZeroValues[_nonZeroValueIndices[j]];
				int row = _rowIndices[j];

				data[i * _rows + row] = value;
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
					printf("%*.*f, ", elementWidth, precision, value);
				}
				else {
					printf("%*.*f", elementWidth, precision, value);
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
	else {
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
					printf("%*.*f, ", elementWidth, precision, value);
				}
				else {
					printf("%*.*f", elementWidth, precision, value);
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
}