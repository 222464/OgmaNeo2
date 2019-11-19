#include "SparseMatrix.h"

using namespace ogmaneo;

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

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &data
) {
	_rows = rows;
	_columns = columns;

	_rowRanges.reserve(_rows + 1);
	_rowRanges.push_back(0);

	int nonZeroCountInRow = 0; // Only need to set this to zero once because it's cumulative
	
	for (int row = 0; row < _rows; row++) {
		int rowOffset = row * _columns;

		for (int col = 0; col < _columns; col++) {
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

void SparseMatrix::initT() {
	_columnRanges.resize(_columns + 1, 0);

	_rowIndices.resize(_nonZeroValues.size());

	_nonZeroValueIndices.resize(_nonZeroValues.size());

	// Pattern for T
	int nextIndex;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++)
			_columnRanges[_columnIndices[j]]++;
	}

	// Bring row range array in place using exclusive scan
	int offset = 0;

	for (int i = 0; i < _columns; i++) {
		int temp = _columnRanges[i];

		_columnRanges[i] = offset;

		offset += temp;
	}

	_columnRanges[_columns] = offset;

	std::vector<int> columnOffsets = _columnRanges;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++) {
			int colIndex = _columnIndices[j];

			int nonZeroIndexT = columnOffsets[colIndex];

			_rowIndices[nonZeroIndexT] = i;

			_nonZeroValueIndices[nonZeroIndexT] = j;

			columnOffsets[colIndex]++;
		}
	}
}

float SparseMatrix::multiply(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		sum += _nonZeroValues[j] * in[_columnIndices[j]];

	return sum;
}

float SparseMatrix::multiplyT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		sum += _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];

	return sum;
}

float SparseMatrix::multiplyNoDiagonal(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++) {
		if (row == _columnIndices[j])
			continue;

		sum += _nonZeroValues[j] * in[_columnIndices[j]];
	}

	return sum;
}

float SparseMatrix::multiplyNoDiagonalT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++) {
		if (column == _rowIndices[j])
			continue;

		sum += _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];
	}

	return sum;
}

int SparseMatrix::count(
	int row
) {
	int nextIndex = row + 1;
	
	return _rowRanges[nextIndex] - _rowRanges[row];
}

int SparseMatrix::countT(
	int column
) {
	int nextIndex = column + 1;
	
	return _columnRanges[nextIndex] - _columnRanges[column];
}

float SparseMatrix::total(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		sum += in[_columnIndices[j]];

	return sum;
}

float SparseMatrix::totalT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		sum += in[_rowIndices[j]];

	return sum;
}

void SparseMatrix::deltas(
	const std::vector<float> &in,
	float delta,
	int row
) {
	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] += delta * in[_columnIndices[j]];
}

void SparseMatrix::deltasT(
	const std::vector<float> &in,
	float delta,
	int column
) {
	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		_nonZeroValues[_nonZeroValueIndices[j]] += delta * in[_rowIndices[j]];
}

void SparseMatrix::normalize(
	int row
) {
	int nextIndex = row + 1;

	float sum = 0.0f;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		sum += _nonZeroValues[j] * _nonZeroValues[j];

	float scale = 1.0f / std::max(0.0001f, std::sqrt(sum));

	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] *= scale;
}

void SparseMatrix::normalizeT(
	int column
) {
	int nextIndex = column + 1;

	float sum = 0.0f;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		sum += _nonZeroValues[_nonZeroValueIndices[j]] * _nonZeroValues[_nonZeroValueIndices[j]];

	float scale = 1.0f / std::max(0.0001f, std::sqrt(sum));

	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		_nonZeroValues[_nonZeroValueIndices[j]] *= scale;
}

float SparseMatrix::magnitude2(
	int row
) {
	int nextIndex = row + 1;

	float sum = 0.0f;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		sum += _nonZeroValues[j] * _nonZeroValues[j];

	return sum;
}

float SparseMatrix::magnitude2T(
	int column
) {
	int nextIndex = column + 1;

	float sum = 0.0f;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		sum += _nonZeroValues[_nonZeroValueIndices[j]] * _nonZeroValues[_nonZeroValueIndices[j]];

	return sum;
}

void SparseMatrix::copyRow(
	const SparseMatrix &source,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] = source._nonZeroValues[j];
}

void SparseMatrix::copyColumn(
	const SparseMatrix &source,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		_nonZeroValues[j] = source._nonZeroValues[j];
}

void SparseMatrix::hebb(
	const std::vector<float> &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] += alpha * (in[_columnIndices[j]] - _nonZeroValues[j]);
}

void SparseMatrix::hebbT(
	const std::vector<float> &in,
	int column,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		_nonZeroValues[_nonZeroValueIndices[j]] += alpha * (in[_rowIndices[j]] - _nonZeroValues[_nonZeroValueIndices[j]]);
}