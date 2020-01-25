#include "SparseMatrix.h"

using namespace ogmaneo;

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) {
	rows = rows;
	columns = columns;

	this->nonZeroValues = nonZeroValues;
	this->rowRanges = rowRanges;
	this->columnIndices = columnIndices;
}

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &data
) {
	rows = rows;
	columns = columns;

	rowRanges.reserve(rows + 1);
	rowRanges.push_back(0);

	int nonZeroCountInRow = 0; // Only need to set this to zero once because it's cumulative
	
	for (int row = 0; row < rows; row++) {
		int rowOffset = row * columns;

		for (int col = 0; col < columns; col++) {
			int index = rowOffset + col;

			if (data[index] != 0.0f) {
				nonZeroValues.push_back(data[index]);
				columnIndices.push_back(col);

				nonZeroCountInRow++;
			}
		}

		rowRanges.push_back(nonZeroCountInRow);
	}
}

void SparseMatrix::initT() {
	columnRanges.resize(columns + 1, 0);

	rowIndices.resize(nonZeroValues.size());

	nonZeroValueIndices.resize(nonZeroValues.size());

	// Pattern for T
	int nextIndex;

	for (int i = 0; i < rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++)
			columnRanges[columnIndices[j]]++;
	}

	// Bring row range array in place using exclusive scan
	int offset = 0;

	for (int i = 0; i < columns; i++) {
		int temp = columnRanges[i];

		columnRanges[i] = offset;

		offset += temp;
	}

	columnRanges[columns] = offset;

	std::vector<int> columnOffsets = columnRanges;

	for (int i = 0; i < rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++) {
			int colIndex = columnIndices[j];

			int nonZeroIndexT = columnOffsets[colIndex];

			rowIndices[nonZeroIndexT] = i;

			nonZeroValueIndices[nonZeroIndexT] = j;

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
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * in[columnIndices[j]];

	return sum;
}

float SparseMatrix::multiplyT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]] * in[rowIndices[j]];

	return sum;
}

float SparseMatrix::multiplyNoDiagonal(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++) {
		if (row == columnIndices[j])
			continue;

		sum += nonZeroValues[j] * in[columnIndices[j]];
	}

	return sum;
}

float SparseMatrix::multiplyNoDiagonalT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++) {
		if (column == rowIndices[j])
			continue;

		sum += nonZeroValues[nonZeroValueIndices[j]] * in[rowIndices[j]];
	}

	return sum;
}

int SparseMatrix::count(
	int row
) {
	int nextIndex = row + 1;
	
	return rowRanges[nextIndex] - rowRanges[row];
}

int SparseMatrix::countT(
	int column
) {
	int nextIndex = column + 1;
	
	return columnRanges[nextIndex] - columnRanges[column];
}

float SparseMatrix::total(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += in[columnIndices[j]];

	return sum;
}

float SparseMatrix::totalT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += in[rowIndices[j]];

	return sum;
}

void SparseMatrix::deltas(
	const std::vector<float> &in,
	float delta,
	int row
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += delta * in[columnIndices[j]];
}

void SparseMatrix::deltasT(
	const std::vector<float> &in,
	float delta,
	int column
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += delta * in[rowIndices[j]];
}

void SparseMatrix::normalize(
	int row
) {
	int nextIndex = row + 1;

	float sum = 0.0f;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * nonZeroValues[j];

	float scale = 1.0f / std::max(0.0001f, std::sqrt(sum));

	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] *= scale;
}

void SparseMatrix::normalizeT(
	int column
) {
	int nextIndex = column + 1;

	float sum = 0.0f;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]] * nonZeroValues[nonZeroValueIndices[j]];

	float scale = 1.0f / std::max(0.0001f, std::sqrt(sum));

	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] *= scale;
}

float SparseMatrix::magnitude2(
	int row
) {
	int nextIndex = row + 1;

	float sum = 0.0f;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * nonZeroValues[j];

	return sum;
}

float SparseMatrix::magnitude2T(
	int column
) {
	int nextIndex = column + 1;

	float sum = 0.0f;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]] * nonZeroValues[nonZeroValueIndices[j]];

	return sum;
}

void SparseMatrix::copyRow(
	const SparseMatrix &source,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] = source.nonZeroValues[j];
}

void SparseMatrix::copyColumn(
	const SparseMatrix &source,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[j] = source.nonZeroValues[j];
}

void SparseMatrix::hebb(
	const std::vector<float> &in,
	int row,
	float post,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += alpha * post * (in[columnIndices[j]] - post * nonZeroValues[j]);
}

void SparseMatrix::hebbT(
	const std::vector<float> &in,
	int column,
	float post,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += alpha * post * (in[rowIndices[j]] - post * nonZeroValues[nonZeroValueIndices[j]]);
}