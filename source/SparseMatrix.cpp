#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) :
	_rows(rows),
	_columns(columns),
	_nonZeroValues(nonZeroValues),
	_rowRanges(rowRanges),
	_columnIndices(columnIndices)
{}

SparseMatrix::SparseMatrix(
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