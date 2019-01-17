#ifndef SPARSE_MATRIX_HEADER
#define SPARSE_MATRIX_HEADER

#include <vector>

// Compressed sparse row (CSR) format
class SparseMatrix {
public:
	std::vector<float> _noneZeroValues;
	std::vector<int> _rowRanges;
	std::vector<int> _columnIndices;

	SparseMatrix() {}
	SparseMatrix(const std::vector<float> &noneZeroValues, const std::vector<int> &rowRanges, const std::vector<int> &columnIndices);
	SparseMatrix(const std::vector<float> &data, int rows, int columns); // From a non-compressed sparse matrix
	void init(const std::vector<float> &noneZeroValues, const std::vector<int> &rowRanges, const std::vector<int> &columnIndices); // If you don't want to construct immediately

	// Size of "in" must equal size of "out"
	void multiply(const std::vector<float> &in, std::vector<float> &out);
	// The range specifies which elements of "out" are to be computed (or which rows of this sparse matrix are to be inner producted with "in")
	void multiplyRange(const std::vector<float> &in, std::vector<float> &out, int startIndex, int length);
	// Treats "in" as having two dimensions
	void multiplyRectangularRange(const std::vector<float> &in, std::vector<float> &out, int startRow, int startColumn, int rectRows, int rectColumns, int rows, int columns);

	// Multiply by a one-hot-row matrix
	// Size of "out" is noneZeroIndicies.size() * columns, which is rows * columns (dense, containing sums)
	void multiplyOHERM(const std::vector<int> &noneZeroIndices, int columns, std::vector<float> &out); // Note that columns is the column size in your view (they are called rows here, not columns, think of the noneZeroIndices as being an Nx1 (tall))
	// Multiply one row from a given one-hot-row matrix
	void multiplyOneRowOHERM(const std::vector<int> &noneZeroIndices, int row, int columns, std::vector<float> &out);
	// Multiply a range of rows from a given one-hot-row matrix
	void multiplyRangeOfRowOHERM(const std::vector<int> &noneZeroIndices, int startRow, int rowCount, int columns, std::vector<float> &out);
};

#endif