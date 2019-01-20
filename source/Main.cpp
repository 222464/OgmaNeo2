#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <unordered_set>

#include "SparseMatrix.h"

class Matrix
{
public:
	std::vector<float> _data;
	int _rows;
	int _columns;

	Matrix() {};
	Matrix(int rows, int columns) : _rows(rows), _columns(columns) {
		_data.resize(_rows * _columns);
	}

	void multiply(const std::vector<float> &in, std::vector<float> &out) {
		for (int row = 0; row < _rows; ++row) {
			int row_offset = row * _columns;

			for (int col = 0; col < _columns; ++col) {
				out[row] += _data[row_offset + col] * in[col];
			}
		}
	}

	void print(int elementWidth) {
		printf("[");
		for (int row = 0; row < _rows; ++row) {
			int row_offset = row * _columns;

			if (row > 0) {
				printf(" [");
			}
			else {
				printf("[");
			}
			for (int col = 0; col < _columns; ++col) {
				float value = _data[row_offset + col];

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

	float &operator()(int row, int column) {
		return _data[row * _columns + column];
	}
};

void printVector(const std::vector<float> &vec, int elementWidth) {
	printf("[");
	for (int i = 0; i < vec.size(); ++i) {
		float value = vec[i];

		if (i < vec.size() - 1) {
			printf("%*.2f, ", elementWidth, value);
		}
		else {
			printf("%*.2f", elementWidth, value);
		}
	}
	printf("]\n");
}

int main() {
	Matrix m(4, 4);
	m(1, 0) = 5;
	m(1, 1) = 8;
	m(2, 2) = 3;
	m(3, 1) = 6;

	SparseMatrix sm(m._data, m._rows, m._columns);

	m.print(6);
	printf("         X\n");

	{
		std::vector<float> test = { 1.0f, 1.0f, 1.0f, 1.0f };
		std::vector<float> result;
		result.resize(test.size(), 0.0f);
		std::vector<float> result2;
		result2.resize(test.size(), 0.0f);

		printVector(test, 6);
		printf("=\n\n");

		m.multiply(test, result);
		sm.multiplyVector(test, result2);

		printf("Normal matrix:\n");
		printVector(result, 6);
		printf("Sparse matrix:\n");
		printVector(result2, 6);
	}

	getchar();
	return 0;
}