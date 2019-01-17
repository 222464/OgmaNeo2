#include <stdlib.h>
#include <stdio.h>

#include <vector>

#include "SparseMatrix.h"
#include "SparseCoder.h"
#include "Helpers.h"

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

using namespace ogmaneo;

struct VisibleLayerDesc {
	/*!
	\brief Visible layer size
	*/
	Int3 _size;

	/*!
	\brief Radius onto hidden layer
	*/
	int _radius;

	/*!
	\brief Initialize defaults
	*/
	VisibleLayerDesc()
		: _size({ 4, 4, 16 }),
		_radius(2)
	{}
};

struct VisibleLayer {
	//!@{
	/*!
	\brief Visible layer values and buffers
	*/
	FloatBuffer _weights;

	FloatBuffer _visibleActivations;

	Float2 _visibleToHidden; // For projection
	Float2 _hiddenToVisible; // For projection

	Int2 _reverseRadii; // Pre-computed reverse radii
						//!@}
};

int main() {
	Matrix m(4, 4);
	m(1, 0) = 5;
	m(1, 1) = 8;
	m(2, 2) = 3;
	m(3, 1) = 6;

	m.print(6);
	printf("\n");

	SparseMatrix sm(m._data, m._rows, m._columns);

	{
		std::vector<float> test = { 1.0f, 1.0f, 1.0f, 1.0f };
		std::vector<float> result;
		result.resize(test.size(), 0.0f);
		std::vector<float> result2;
		result2.resize(test.size(), 0.0f);

		m.multiply(test, result);
		sm.multiply(test, result2);

		printf("Normal matrix:\n");
		printVector(result, 6);
		printf("Sparse matrix:\n");
		printVector(result2, 6);
	} 

	{
		// Pulled from SparseCoder.cpp

		// Test values set by me
		Int2 pos;
		Int3 _hiddenSize;
		std::vector<VisibleLayer> _visibleLayers;
		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		FloatBuffer _hiddenActivations;

		// Cache address calculations
		int dxy = _hiddenSize.x * _hiddenSize.y;
		int dxyz = dxy * _hiddenSize.z;

		// Running max data
		int maxIndex = 0;
		float maxValue = -999999.0f;

		// For each hidden cell
		for (int hc = 0; hc < _hiddenSize.z; hc++) {
			Int3 hiddenPosition(pos.x, pos.y, hc);

			// Partial sum cache value
			int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

			// Accumulator
			float sum = 0.0f;

			// For each visible layer
			for (int vli = 0; vli < _visibleLayers.size(); vli++) {
				VisibleLayer &vl = _visibleLayers[vli];
				VisibleLayerDesc &vld = _visibleLayerDescs[vli];

				Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

				// Lower corner
				Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

				// Additional addressing dimensions
				int diam = vld._radius * 2 + 1;
				int diam2 = diam * diam;

				// Bounds of receptive field, clamped to input size
				Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
				Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

				for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
					for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
						Int2 visiblePosition(x, y);

						int visibleIndex = address2(visiblePosition, vld._size.x);

						int visibleC = (*inputCs[vli])[visibleIndex];

						// Complete the partial address with final value needed
						int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

						// Rule is: sum += max(0, weight - prevActivation), found empirically to be better than truncated weight * (1.0 - prevActivation) update
						sum += vl._weights[dPartial + az * dxyz];
					}
			}

			int hiddenIndex = address3(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y));

			_hiddenActivations[hiddenIndex] = sum;
		}

		// Same thing, but with SparseMatrix


		// Compare the sums
	}

	getchar();
	return 0;
}