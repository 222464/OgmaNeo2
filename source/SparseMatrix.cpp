#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) :
	_nonZeroValues(nonZeroValues),
	_rowRanges(rowRanges),
	_columnIndices(columnIndices)
{}

SparseMatrix::SparseMatrix(
	const std::vector<float> &data,
	int rows,
	int columns
) {
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

void SparseMatrix::init(
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) {
	_nonZeroValues = nonZeroValues;
	_rowRanges = rowRanges;
	_columnIndices = columnIndices;
}

void SparseMatrix::multiply(
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

void SparseMatrix::multiplyRange(
	const std::vector<float> &in,
	std::vector<float> &out,
	int startIndex,
	int length
) {
	int end = startIndex + length;

	int nextIndex = 0;
	for (int i = startIndex; i < end; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; ++j) {
			out[i] += _nonZeroValues[j] * in[_columnIndices[j]];
		}
	}
}

void SparseMatrix::multiplyRectangularRange(
	const std::vector<float> &in,
	std::vector<float> &out,
	int startRow,
	int startColumn,
	int rectRows,
	int rectColumns,
	int rows,
	int columns
) {
	if (startRow < 0)
		startRow = 0;
	else if (startRow >= rows)
		return;

	if (startColumn < 0)
		startColumn = 0;
	else if (startColumn >= columns)
		return;
	
	int rowEnd = startRow + rectRows;
	int columnEnd = startColumn + rectColumns;

	if (rowEnd >= rows) rowEnd = rows - 1;
	if (columnEnd >= columns) columnEnd = columns - 1;

	for (int row = startRow; row < rowEnd; ++row) {
		int row_offset = row * columns;
		for (int col = startColumn; col < columnEnd; ++col) {
			int index = row_offset + col;
			int nextIndex = index + 1;

			for (int j = _rowRanges[index]; j < _rowRanges[nextIndex]; ++j) {
				out[index] += _nonZeroValues[j] * in[_columnIndices[j]];
			}
		}
	}
}

void SparseMatrix::multiplyOHERM(
	const std::vector<int> &nonZeroIndices,
	int columns,
	std::vector<float> &out
) {
	int rows = nonZeroIndices.size();

	for (int row = 0; row < rows; ++row) {
		int index = row * columns + nonZeroIndices[row];
		int nextIndex = index + 1;

		for (int j = _rowRanges[index]; j < _rowRanges[nextIndex]; ++j) {
			out[index] += _nonZeroValues[j];
		}
	}
}

void SparseMatrix::multiplyOneRowOHERM(
	const std::vector<int> &nonZeroIndices,
	int row, int columns,
	std::vector<float> &out
) {
	int index = row * columns + nonZeroIndices[row];
	int nextIndex = index + 1;

	for (int j = _rowRanges[index]; j < _rowRanges[nextIndex]; ++j) {
		out[index] += _nonZeroValues[j];
	}
}

void SparseMatrix::multiplyRangeOfRowOHERM(
	const std::vector<int> &nonZeroIndices,
	int startRow,
	int rowCount,
	int columns,
	std::vector<float> &out
) {
	int rowEnd = startRow + rowCount;

	for (int row = startRow; row < rowEnd; ++row) {
		int index = row * columns + nonZeroIndices[row];
		int nextIndex = index + 1;

		for (int j = _rowRanges[index]; j < _rowRanges[nextIndex]; ++j) {
			out[index] += _nonZeroValues[j];
		}
	}
}