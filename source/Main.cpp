#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <unordered_set>

#include "SparseMatrix.h"
#include "SparseCoder.h"
#include "Helpers.h"

#include <SFML/Graphics.hpp>

#include <iostream>

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
	std::mt19937 rng(time(nullptr));

	Int3 hiddenSize(6, 6, 8);
	Int3 inputSize(6, 6, 16);

	int rows = hiddenSize.x * hiddenSize.y * hiddenSize.z;
	int columns = inputSize.x * inputSize.y * inputSize.z;

	SparseMatrix m;

	initSMLocalRF(inputSize, hiddenSize, 1, m);

	m.initT();

	// Regular
	{
		sf::Image img;

		img.create(columns, rows);

		int nextIndex;

		std::uniform_int_distribution<int> colorDist(200, 255);

		int total = 0;
		
		for (int i = 0; i < rows; i = nextIndex) {
			nextIndex = i + 1;

			for (int j = m._rowRanges[i]; j < m._rowRanges[nextIndex]; j++) {
				sf::Color c = sf::Color::White;

				c.r = colorDist(rng);
				c.g = colorDist(rng);
				c.b = colorDist(rng);

				img.setPixel(m._columnIndices[j], i, c);

				total++;
			}
		}

		std::cout << "Total: " << total << std::endl;

		img.saveToFile("sm.png");
	}

	// T
	{
		sf::Image img;

		img.create(rows, columns);

		int nextIndex;

		std::uniform_int_distribution<int> colorDist(200, 255);

		int total = 0;
		
		for (int i = 0; i < columns; i = nextIndex) {
			nextIndex = i + 1;

			for (int j = m._columnRanges[i]; j < m._columnRanges[nextIndex]; j++) {
				sf::Color c = sf::Color::White;

				c.r = colorDist(rng);
				c.g = colorDist(rng);
				c.b = colorDist(rng);

				img.setPixel(m._rowIndices[j], i, c);

				total++;
			}
		}

		std::cout << "TotalT: " << total << std::endl;

		img.saveToFile("smT.png");
	}

	std::cout << "Saved." << std::endl;

	// Matrix m(4, 4);
	// m(1, 0) = 5;
	// m(1, 1) = 8;
	// m(2, 2) = 3;
	// m(3, 1) = 6;

	// m.print(6);
	// printf("\n");

	// SparseMatrix sm(4, 4, m._data);

	// {
	// 	std::vector<float> test = { 1.0f, 1.0f, 1.0f, 1.0f };
	// 	std::vector<float> result;
	// 	result.resize(test.size(), 0.0f);
	// 	std::vector<float> result2;
	// 	result2.resize(test.size(), 0.0f);

	// 	m.multiply(test, result);
	// 	sm.multiply(test, result2);

	// 	printf("Normal matrix:\n");
	// 	printVector(result, 6);
	// 	printf("Sparse matrix:\n");
	// 	printVector(result2, 6);
	// } 

	// {
	// 	// Pulled from SparseCoder.cpp

	// 	// Test values set by me
	// 	Int2 pos;
	// 	Int3 _hiddenSize(4, 4, 16);
	// 	std::vector<VisibleLayerDesc> _visibleLayerDescs;
	// 	_visibleLayerDescs.reserve(1);
	// 	_visibleLayerDescs[0]._radius = 2;
	// 	_visibleLayerDescs[0]._size = Int3(4, 4, 16);
	// 	std::vector<VisibleLayer> _visibleLayers;
	// 	_visibleLayers.resize(1);
	// 	_visibleLayers[0]._hiddenToVisible = Float2(1.0f, 1.0f);
	// 	_visibleLayers[0]._visibleToHidden = Float2(1.0f, 1.0f);
	// 	{
	// 		_hiddenSize = _hiddenSize;

	// 		_visibleLayers.resize(_visibleLayerDescs.size());

	// 		// Pre-compute dimensions
	// 		int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
	// 		int numHidden = numHiddenColumns * _hiddenSize.z;

	// 		// Create layers
	// 		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
	// 			VisibleLayer &vl = _visibleLayers[vli];
	// 			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

	// 			int numVisibleColumns = vld._size.x * vld._size.y;
	// 			int numVisible = numVisibleColumns * vld._size.z;

	// 			// Projection constants
	// 			vl._visibleToHidden = Float2(static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
	// 				static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y));

	// 			vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
	// 				static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

	// 			vl._reverseRadii = Int2(static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
	// 				static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1));

	// 			int diam = vld._radius * 2 + 1;

	// 			int numWeightsPerHidden = diam * diam * vld._size.z;

	// 			int weightsSize = numHidden * numWeightsPerHidden;

	// 			// Create weight matrix for this visible layer and initialize randomly
	// 			vl._weights.resize(weightsSize, 0.0f);
	// 			for (int i = 0; i < vl._weights.size(); ++i) (float)((rand() % 1000) / 1000.0f);
	// 		}
	// 	}
	// 	std::vector<IntBuffer *> inputCs; // This is what I call the one-hot-row matrix (aka the SDR in this project)
	// 	inputCs.resize(1);
	// 	std::vector<int> OHERM;
	// 	OHERM.resize(_visibleLayerDescs[0]._size.x * _visibleLayerDescs[0]._size.y);
	// 	for (int i = 0; i < OHERM.size(); ++i) OHERM[i] = rand() % _visibleLayerDescs[0]._size.z;
	// 	inputCs[0] = &OHERM;

	// 	FloatBuffer _hiddenActivations;
	// 	_hiddenActivations.resize(_hiddenSize.x * _hiddenSize.y * _hiddenSize.z, 0.0f);

	// 	FloatBuffer myActivations;
	// 	myActivations.resize(_hiddenSize.x * _hiddenSize.y * _hiddenSize.z, 0.0f);

	// 	SparseMatrix sparse;
	// 	sparse._rowRanges.push_back(0);
	// 	//sparse._noneZeroValues.resize(_visibleLayers[0]._weights.size(), 0.0f);
	// 	//sparse._rowRanges.resize(sparse._noneZeroValues.size() + 1, 0);
	// 	//sparse._columnIndices.resize(sparse._noneZeroValues.size(), 0);

	// 	// Construct sparse matrix
	// 	for (int i = 0; i < _hiddenSize.y; ++i) {
	// 		for (int j = 0; j < _hiddenSize.x; ++j) {
	// 			pos.x = j;
	// 			pos.y = i;

	// 			// Cache address calculations
	// 			int dxy = _hiddenSize.x * _hiddenSize.y;
	// 			int dxyz = dxy * _hiddenSize.z;

	// 			// Running max data
	// 			int maxIndex = 0;
	// 			float maxValue = -999999.0f;

	// 			// For each hidden cell
	// 			for (int hc = 0; hc < _hiddenSize.z; hc++) {
	// 				Int3 hiddenPosition(pos.x, pos.y, hc);

	// 				std::vector<int> notZero;

	// 				// Partial sum cache value
	// 				int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

	// 				// Accumulator
	// 				float sum = 0.0f;

	// 				// For each visible layer
	// 				for (int vli = 0; vli < _visibleLayers.size(); vli++) {
	// 					VisibleLayer &vl = _visibleLayers[vli];
	// 					VisibleLayerDesc &vld = _visibleLayerDescs[vli];

	// 					Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

	// 					// Lower corner
	// 					Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

	// 					// Additional addressing dimensions
	// 					int diam = vld._radius * 2 + 1;
	// 					int diam2 = diam * diam;

	// 					// Bounds of receptive field, clamped to input size
	// 					Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
	// 					Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

	// 					for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
	// 						for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
	// 							Int2 visiblePosition(x, y);

	// 							int visibleIndex = address2R(visiblePosition, vld._size.x);

	// 							int visibleC = (*inputCs[vli])[visibleIndex];

	// 							// Complete the partial address with final value needed
	// 							int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

	// 							notZero.push_back(dPartial + az * dxyz);
	// 						}
	// 				}

	// 				for (int p = 0; p < notZero.size(); ++p) {
	// 					sparse._nonZeroValues.push_back(_visibleLayers[0]._weights[notZero[p]]);
	// 					//sparse._columnIndices.push_back(az * dxyz);
	// 				}

	// 				sparse._rowRanges.push_back(sparse._rowRanges[sparse._rowRanges.size() - 1] + notZero.size());
	// 			}
	// 		}
	// 	}

	// 	for (int i = 0; i < _hiddenSize.y; ++i) {
	// 		for (int j = 0; j < _hiddenSize.x; ++j) {
	// 			pos.x = j;
	// 			pos.y = i;

	// 			// Cache address calculations
	// 			int dxy = _hiddenSize.x * _hiddenSize.y;
	// 			int dxyz = dxy * _hiddenSize.z;

	// 			// Running max data
	// 			int maxIndex = 0;
	// 			float maxValue = -999999.0f;

	// 			// For each hidden cell
	// 			for (int hc = 0; hc < _hiddenSize.z; hc++) {
	// 				Int3 hiddenPosition(pos.x, pos.y, hc);

	// 				// Partial sum cache value
	// 				int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

	// 				// Accumulator
	// 				float sum = 0.0f;

	// 				// For each visible layer
	// 				for (int vli = 0; vli < _visibleLayers.size(); vli++) {
	// 					VisibleLayer &vl = _visibleLayers[vli];
	// 					VisibleLayerDesc &vld = _visibleLayerDescs[vli];

	// 					Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

	// 					// Lower corner
	// 					Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

	// 					// Additional addressing dimensions
	// 					int diam = vld._radius * 2 + 1;
	// 					int diam2 = diam * diam;

	// 					// Bounds of receptive field, clamped to input size
	// 					Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
	// 					Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

	// 					for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
	// 						for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
	// 							Int2 visiblePosition(x, y);

	// 							int visibleIndex = address2R(visiblePosition, vld._size.x);

	// 							int visibleC = (*inputCs[vli])[visibleIndex];

	// 							// Complete the partial address with final value needed
	// 							int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

	// 							// Rule is: sum += max(0, weight - prevActivation), found empirically to be better than truncated weight * (1.0 - prevActivation) update
	// 							sum += vl._weights[dPartial + az * dxyz];
	// 						}
	// 				}

	// 				int hiddenIndex = address3R(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y));

	// 				_hiddenActivations[hiddenIndex] = sum;
	// 			}
	// 		}
	// 	}

	// 	// Same thing, but with SparseMatrix


	// 	// Compare the sums
	// }

	// getchar();
	return 0;
}