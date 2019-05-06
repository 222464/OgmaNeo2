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

float SparseMatrix::distance(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++) {
		float delta = in[_columnIndices[j]] - _nonZeroValues[j];

		sum += delta * delta;
	}

	return sum;
}

int SparseMatrix::counts(
	int row
) {
	int nextIndex = row + 1;
	
	return _rowRanges[nextIndex] - _rowRanges[row];
}

float SparseMatrix::counts(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		sum += in[_columnIndices[j]];

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

float SparseMatrix::distanceT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++) {
		float delta = in[_rowIndices[j]] - _nonZeroValues[_nonZeroValueIndices[j]];
	
		sum += delta * delta;
	}

	return sum;
}

int SparseMatrix::countsT(
	int column
) {
	int nextIndex = column + 1;
	
	return _columnRanges[nextIndex] - _columnRanges[column];
}

float SparseMatrix::countsT(
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

float SparseMatrix::min(
	const std::vector<float> &in,
	int row
) {
	float m = 999999.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		m = std::min(m, std::min(_nonZeroValues[j], in[_columnIndices[j]]));

	return m;
}

float SparseMatrix::max(
	const std::vector<float> &in,
	int row
) {
	float m = -999999.0f;

	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		m = std::max(m, std::min(_nonZeroValues[j], in[_columnIndices[j]]));

	return m;
}

float SparseMatrix::minT(
	const std::vector<float> &in,
	int column
) {
	float m = 999999.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		m = std::min(m, std::min(_nonZeroValues[_nonZeroValueIndices[j]], in[_rowIndices[j]]));

	return m;
}

float SparseMatrix::maxT(
	const std::vector<float> &in,
	int column
) {
	float m = -999999.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		m = std::max(m, std::min(_nonZeroValues[_nonZeroValueIndices[j]], in[_rowIndices[j]]));

	return m;
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

void SparseMatrix::scale(
	int row,
	float scale
) {
	int nextIndex = row + 1;

	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] *= scale;
}

void SparseMatrix::scaleT(
	int column,
	float scale
) {
	int nextIndex = column + 1;

	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		_nonZeroValues[_nonZeroValueIndices[j]] *= scale;
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

void SparseMatrix::hebbOHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int jj = _rowRanges[row]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[_columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			_nonZeroValues[j] += alpha * (target - _nonZeroValues[j]);
		}
	}
}

void SparseMatrix::hebbOHVsT(
	const std::vector<int> &nonZeroIndices,
	int column,
	int oneHotSize,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int jj = _columnRanges[column]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[_rowIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			_nonZeroValues[_nonZeroValueIndices[j]] += alpha * (target - _nonZeroValues[_nonZeroValueIndices[j]]);
		}
	}
}

void SparseMatrix::trace(
	const std::vector<float> &in,
	int row
) {
	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] = std::max(_nonZeroValues[j], in[_columnIndices[j]]);
}

void SparseMatrix::traceT(
	const std::vector<float> &in,
	int column
) {
	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		_nonZeroValues[_nonZeroValueIndices[j]] = std::max(_nonZeroValues[_nonZeroValueIndices[j]], in[_rowIndices[j]]);
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