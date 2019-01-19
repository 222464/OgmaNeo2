#include "SparseMatrix.h"

using namespace ogmaneo;

void SparseMatrix::create(
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

void SparseMatrix::create(
	int rows,
	int columns,
	const std::vector<float> &data
) {
	_rows = rows;
	_columns = columns;

	_rowRanges.reserve(rows + 1);
	_rowRanges.push_back(0);

	int nonZeroCountInRow = 0; // Only need to set this to zero once because it's cumulative
	
	for (int row = 0; row < rows; row++) {
		int rowOffset = row * columns;

		for (int col = 0; col < columns; col++) {
			int index = rowOffset + col;

			if (data[index] != 0.0f) {
				_nonZeroValues.push_back(data[index]);
				_columnIndices.push_back(col);
				nonZeroCountInRow++;
			}
		}

		_rowRanges.push_back(nonZeroCountInRow);
	}
}

void SparseMatrix::createT() {
	_rowRangesT.resize(_columns + 1);
	_rowRangesT[0] = 0;

	_columnIndicesT.resize(_nonZeroValues.size());

	// Temporary nonzero value count
	std::vector<int> nonZeroCountInRowT(_columns, 0);

	int nextIndex = 0;

	int nonZeroValueIndex = 0;
	
	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++) {
			_columnIndicesT[nonZeroValueIndex] = i;

			nonZeroCountInRowT[_columnIndices[j]]++;

			nonZeroValueIndex++;
		}
	}

	// Build _rowRangesT
	for (int i = 0; i < nonZeroCountInRowT.size(); i++)
		_rowRangesT[i + 1] = _rowRangesT[i] + nonZeroCountInRowT[i];
}

void SparseMatrix::multiply(
	const std::vector<float> &in,
	std::vector<float> &out
) {
	assert(in.size() == _rows);

	int nextIndex = 0;
	
	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++) {
			out[i] += _nonZeroValues[j] * in[_columnIndices[j]];
		}
	}
}

void SparseMatrix::multiplyRange(
	const std::vector<float> &in,
	std::vector<float> &out,
	int startRow,
	int rowCount
) {
	int end = startRow + rowCount;

	int nextIndex = 0;

	for (int i = startRow; i < end; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++)
			out[i] += _nonZeroValues[j] * in[_columnIndices[j]];
	}
}

void SparseMatrix::multiplyOHVs(
	const std::vector<int> &nonZeroIndices,
	std::vector<float> &out
) {
	assert(nonZeroIndices.size() == _rows);

	for (int row = 0; row < _rows; row++) {
		int index = row * _columns + nonZeroIndices[row];
		int nextIndex = index + 1;

		for (int j = _rowRanges[index]; j < _rowRanges[nextIndex]; j++)
			out[index] += _nonZeroValues[j];
	}
}

void SparseMatrix::multiplyRangeOfRowOHVs(
	const std::vector<int> &nonZeroIndices,
	int startRow,
	int rowCount,
	std::vector<float> &out
) {
	int rowEnd = startRow + rowCount;

	for (int row = startRow; row < rowEnd; row++) {
		int index = row * _columns + nonZeroIndices[row];
		int nextIndex = index + 1;

		for (int j = _rowRanges[index]; j < _rowRanges[nextIndex]; j++)
			out[index] += _nonZeroValues[j];
	}
}