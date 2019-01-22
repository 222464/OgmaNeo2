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

void SparseMatrix::multiply(
	const std::vector<float> &in,
	std::vector<float> &out
) {
	int nextIndex;
	
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

	int nextIndex;

	for (int i = startRow; i < end; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++)
			out[i] += _nonZeroValues[j] * in[_columnIndices[j]];
	}
}

void SparseMatrix::counts(
	std::vector<int> &out
) {
	int nextIndex;
	
	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		out[i] += _rowRanges[nextIndex] - _rowRanges[i];
	}
}

void SparseMatrix::multiplyT(
	const std::vector<float> &in,
	std::vector<float> &out
) {
	assert(in.size() == _rows);

	int nextIndex;
	
	for (int i = 0; i < _columns; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; j++) {
			out[i] += _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];
		}
	}
}

void SparseMatrix::multiplyRangeT(
	const std::vector<float> &in,
	std::vector<float> &out,
	int startColumn,
	int columnCount
) {
	int endColumn = startColumn + columnCount;

	int nextIndex;

	for (int i = startColumn; i < endColumn; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _columnRanges[i]; j < _columnRanges[nextIndex]; j++)
			out[i] += _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];
	}
}

void SparseMatrix::countsT(
	std::vector<int> &out
) {
	int nextIndex;
	
	for (int i = 0; i < _columns; i = nextIndex) {
		nextIndex = i + 1;

		out[i] += _columnRanges[nextIndex] - _columnRanges[i];
	}
}

void SparseMatrix::multiplyOHVs(
	const std::vector<int> &nonZeroIndices,
	std::vector<float> &out,
	int oneHotSize,
	bool negative
) {
	if (negative) {
		int nextIndex;
		
		for (int i = 0; i < _rows; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _rowRanges[i]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

				out[i] -= _nonZeroValues[j];
			}
		}
	}
	else {
		int nextIndex;
		
		for (int i = 0; i < _rows; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _rowRanges[i]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

				out[i] += _nonZeroValues[j];
			}
		}
	}
}

void SparseMatrix::multiplyRangeOHVs(
	const std::vector<int> &nonZeroIndices,
	std::vector<float> &out,
	int startRow,
	int rowCount,
	int oneHotSize,
	bool negative
) {
	int endRow = startRow + rowCount;

	if (negative) {
		int nextIndex;
		
		for (int i = startRow; i < endRow; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _rowRanges[i]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

				out[i] -= _nonZeroValues[j];
			}
		}
	}
	else {
		int nextIndex;
		
		for (int i = startRow; i < endRow; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _rowRanges[i]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

				out[i] += _nonZeroValues[j];
			}
		}
	}
}

void SparseMatrix::multiplyOHVsT(
	const std::vector<int> &nonZeroIndices,
	std::vector<float> &out,
	int oneHotSize,
	bool negative
) {
	if (negative) {
		int nextIndex;
		
		for (int i = 0; i < _columns; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _columnRanges[i]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

				out[i] -= _nonZeroValues[_nonZeroValueIndices[j]];
			}
		}
	}
	else {
		int nextIndex;
		
		for (int i = 0; i < _columns; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _columnRanges[i]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

				out[i] += _nonZeroValues[_nonZeroValueIndices[j]];
			}
		}
	}
}

void SparseMatrix::multiplyRangeOHVsT(
	const std::vector<int> &nonZeroIndices,
	std::vector<float> &out,
	int startColumn,
	int columnCount,
	int oneHotSize,
	bool negative
) {
	int endColumn = startColumn + columnCount;

	if (negative) {
		int nextIndex;
		
		for (int i = startColumn; i < endColumn; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _columnRanges[i]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

				out[i] -= _nonZeroValues[_nonZeroValueIndices[j]];
			}
		}
	}
	else {
		int nextIndex;
		
		for (int i = startColumn; i < endColumn; i = nextIndex) {
			nextIndex = i + 1;

			for (int jj = _columnRanges[i]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
				int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

				out[i] += _nonZeroValues[_nonZeroValueIndices[j]];
			}
		}
	}
}

void SparseMatrix::deltaRuleRangeOHVs(
	const std::vector<int> &nonZeroIndices,
	const std::vector<float> &deltas,
	int startRow,
	int rowCount,
	int oneHotSize
) {
	int endRow = startRow + rowCount;

	int nextIndex;
	
	for (int i = startRow; i < endRow; i = nextIndex) {
		nextIndex = i + 1;

		for (int jj = _rowRanges[i]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
			int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

			_nonZeroValues[j] += deltas[i];
		}
	}
}

void SparseMatrix::deltaOHVRuleOHVs(
	const std::vector<int> &nonZeroIndices,
	int OHVIndex,
	int inputOneHotSize,
	int outputOneHotSize,
	int positiveIndex,
	int negativeIndex,
	float alpha
) {
	if (positiveIndex == negativeIndex)
		return;

	int startIndex = OHVIndex * outputOneHotSize;

	// Positive	
	{
		int positiveRow = startIndex + positiveIndex;

		int nextIndex = positiveRow + 1;
		
		for (int jj = _rowRanges[positiveRow]; jj < _rowRanges[nextIndex]; jj += inputOneHotSize) {
			int j = jj + nonZeroIndices[_columnIndices[jj] / inputOneHotSize];

			_nonZeroValues[j] += alpha;
		}
	}

	// Negative	
	{
		int negativeRow = startIndex + negativeIndex;

		int nextIndex = negativeRow + 1;
		
		for (int jj = _rowRanges[negativeRow]; jj < _rowRanges[nextIndex]; jj += inputOneHotSize) {
			int j = jj + nonZeroIndices[_columnIndices[jj] / inputOneHotSize];

			_nonZeroValues[j] -= alpha;
		}
	}
}

void SparseMatrix::deltaRuleRangeOHVsT(
	const std::vector<int> &nonZeroIndices,
	const std::vector<float> &deltas,
	int startColumn,
	int columnCount,
	int oneHotSize
) {
	int endColumn = startColumn + columnCount;

	int nextIndex;
	
	for (int i = startColumn; i < endColumn; i = nextIndex) {
		nextIndex = i + 1;

		for (int jj = _columnRanges[i]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
			int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

			_nonZeroValues[_nonZeroValueIndices[j]] += deltas[i];
		}
	}
}

void SparseMatrix::deltaOHVRuleOHVsT(
	const std::vector<int> &nonZeroIndices,
	int OHVIndex,
	int inputOneHotSize,
	int outputOneHotSize,
	int positiveIndex,
	int negativeIndex,
	float alpha
) {
	if (positiveIndex == negativeIndex)
		return;

	int startIndex = OHVIndex * outputOneHotSize;

	// Positive	
	{
		int positiveRow = startIndex + positiveIndex;

		int nextIndex = positiveRow + 1;
		
		for (int jj = _columnRanges[positiveRow]; jj < _columnRanges[nextIndex]; jj += inputOneHotSize) {
			int j = jj + nonZeroIndices[_rowIndices[jj] / inputOneHotSize];

			_nonZeroValues[_nonZeroValueIndices[j]] += alpha;
		}
	}

	// Negative	
	{
		int negativeRow = startIndex + negativeIndex;

		int nextIndex = negativeRow + 1;
		
		for (int jj = _columnRanges[negativeRow]; jj < _columnRanges[nextIndex]; jj += inputOneHotSize) {
			int j = jj + nonZeroIndices[_rowIndices[jj] / inputOneHotSize];

			_nonZeroValues[_nonZeroValueIndices[j]] -= alpha;
		}
	}
}

void SparseMatrix::hebbRuleDecreasing(
	const std::vector<float> &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] += alpha * std::min(0.0f, in[_columnIndices[j]] - _nonZeroValues[j]);
}

void SparseMatrix::hebbRuleDecreasingOHVs(
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

			_nonZeroValues[j] += alpha * std::min(0.0f, target - _nonZeroValues[j]);
		}
	}
}