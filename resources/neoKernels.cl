// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ------------------------------------------- Common -------------------------------------------

// MWC64X
uint rand(
    uint2 *state
) {
    enum { A = 4294883355U };

    uint x = (*state).x, c = (*state).y;
    
    uint res = x ^ c;

    uint hi = mul_hi(x, A);

    x = x * A + c;

    c = hi + (x < c);

    *state = (uint2)(x, c);

    return res;
}

float randFloat(
    uint2* state
) {
    uint tmp = rand(state);

    return (tmp % 99999) / 99999.0f;
}

float randNormal(
    uint2* state
) {
    float u1 = randFloat(state);
    float u2 = randFloat(state);

    return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
}

bool inBounds0(
    int2 position,
    int2 upperBound
) {
    return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
}

bool inBounds(
    int2 position,
    int2 lowerBound,
    int2 upperBound
) {
    return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
}

int address2(
    int2 pos,
    int2 dim
) {
    return pos.y + pos.x * dim.y;
}

int address3(
    int3 pos,
    int3 dims
) {
    return pos.z + pos.y * dims.z + pos.x * dims.y * dims.z;
}

int address4(
    int4 pos,
    int4 dims
) {
    return pos.w + pos.z * dims.w + pos.y * dims.z * dims.w + pos.x * dims.y * dims.z * dims.w;
}

inline float sigmoid(
    float x
) {
    if (x < 0.0f) {
        x = exp(x);

        return x / (1.0f + x);
    }
    
    return 1.0f / (1.0f + exp(-x));
}

float multiplyOHVs(
    global const float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    global const int* nonZeroIndices,
    int row,
    int oneHotSize
) {
    float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		sum += nonZeroValues[j];
	}

    return sum;
}

float multiplyOHVsT(
    global const float* nonZeroValues,
    global const int* columnRanges,
    global const int* rowIndices,
    global const int* nonZeroValueIndices,
    global const int* nonZeroIndices,
    int column,
    int oneHotSize
) {
    float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		sum += nonZeroValues[nonZeroValueIndices[j]];
	}

    return sum;
}

void deltaOHVs(
    global float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    global const int* nonZeroIndices,
    float delta,
    int row,
    int oneHotSize
) {
	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		nonZeroValues[j] += delta;
	}
}

void deltaOHVsT(
    global float* nonZeroValues,
    global const int* columnRanges,
    global const int* rowIndices,
    global const int* nonZeroValueIndices,
    global const int* nonZeroIndices,
    float delta,
    int column,
    int oneHotSize
) {
    int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		nonZeroValues[nonZeroValueIndices[j]] += delta;
    }
}

float multiply(
	global float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    global const float* inputs,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * inputs[columnIndices[j]];

	return sum;
}

// void deltasT(
//     global float* nonZeroValues,
//     global const int* columnRanges,
//     global const int* rowIndices,
//     global const int* nonZeroValueIndices,
//     global const float* inputs,
//     float delta,
//     int column
// ) {
//     int nextIndex = column + 1;
	
// 	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
//         nonZeroValues[nonZeroValueIndices[j]] += delta * inputs[rowIndices[j]];
// }

int counts(
    global const int* rowRanges,
	int row
) {
	int nextIndex = row + 1;
	
	return rowRanges[nextIndex] - rowRanges[row];
}

int countsT(
    global const int* columnRanges,
	int column
) {
	int nextIndex = column + 1;
	
	return columnRanges[nextIndex] - columnRanges[column];
}

// ------------------------------------------- Sparse Coder -------------------------------------------

void kernel scForward(
    global const int* visibleCs,
    global float* hiddenActivations,
    global const float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    int3 visibleSize,
    int3 hiddenSize
) {
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int hiddenIndex = address3(hiddenPosition, hiddenSize);

    hiddenActivations[hiddenIndex] += multiplyOHVs(nonZeroValues, rowRanges, columnIndices, visibleCs, hiddenIndex, visibleSize.z);
}

void kernel scInhibit(
    global const float* hiddenActivations,
    global int* hiddenCs,
    int3 hiddenSize
) {
    int2 hiddenColumnPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = -999999.0f;
    
    // Find max
    for (int c = 0; c < hiddenSize.z; c++) {
        int hiddenIndex = address3((int3)(hiddenColumnPosition, c), hiddenSize);

        float value = hiddenActivations[hiddenIndex];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenColumnPosition, hiddenSize.xy)] = maxIndex;
}

void kernel scLearn(
    global const int* visibleCs,
    global const int* hiddenCs,
    global float* nonZeroValues,
    global const int* nonZeroValueIndices,
    global const int* columnRanges,
    global const int* rowIndices,
    int3 visibleSize,
    int3 hiddenSize,
    float alpha
) {
    int3 visiblePosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int visibleColumnIndex = address2(visiblePosition.xy, visibleSize.xy);

    int visibleC = visibleCs[visibleColumnIndex];
    
    int visibleIndex = address3(visiblePosition, visibleSize);

    float sum = multiplyOHVsT(nonZeroValues, columnRanges, rowIndices, nonZeroValueIndices, hiddenCs, visibleIndex, hiddenSize.z);

    sum /= (countsT(columnRanges, visibleColumnIndex * visibleSize.z) / hiddenSize.z);

    float delta = alpha * ((visiblePosition.z == visibleC ? 1.0f : 0.0f) - sum);

    deltaOHVsT(nonZeroValues, columnRanges, rowIndices, nonZeroValueIndices, hiddenCs, delta, visibleIndex, hiddenSize.z);
}

// ------------------------------------------- Actor -------------------------------------------

void kernel aCount(
    global const int* rowRanges,
    global int* hiddenCounts,
    int3 visibleSize,
    int3 hiddenSize
) {
    int2 hiddenColumnPosition = (int2)(get_global_id(0), get_global_id(1));
	      
    int hiddenColumnIndex = address2(hiddenColumnPosition, hiddenSize.xy);

    hiddenCounts[hiddenColumnIndex] += counts(rowRanges, hiddenColumnIndex * (hiddenSize.z + 1)) / visibleSize.z;
}

void kernel aForward(
    global const int* visibleCs,
    global float* hiddenActivations,
    global const float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    int3 visibleSize,
    int3 hiddenSize
) {
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int hiddenIndex = address3(hiddenPosition, hiddenSize);
	      
    hiddenActivations[hiddenIndex] += multiplyOHVs(nonZeroValues, rowRanges, columnIndices, visibleCs, hiddenIndex, visibleSize.z);
}

void kernel aInhibit(
    global const float* hiddenActivations,
    global int* hiddenCs,
    int3 hiddenSize,
    float epsilon,
    uint2 seed
) {
    int2 hiddenColumnPosition = (int2)(get_global_id(0), get_global_id(1));

    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    int selectIndex = 0;

    if (randFloat(&stateValue) < epsilon)
        selectIndex = (int)(rand(&stateValue) % hiddenSize.z);
    else {
        float maxValue = hiddenActivations[address3((int3)(hiddenColumnPosition, 0), hiddenSize)];
    
        // Find max
        for (int c = 1; c < hiddenSize.z; c++) {
            float value = hiddenActivations[address3((int3)(hiddenColumnPosition, c), hiddenSize)];

            if (value > maxValue) {
                maxValue = value;
                selectIndex = c;
            }
        }
    }

    // Set states
    hiddenCs[address2(hiddenColumnPosition, hiddenSize.xy)] = selectIndex;
}

void kernel aLearn(
    global const int* visibleCsPrev,
    global const float* hiddenActivations,
    global const float* hiddenActivationsPrev,
    global const int* hiddenCsPrev,
    global const int* hiddenCounts,
    global float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    int3 visibleSize,
    int3 hiddenSize,
    float alpha,
    float gamma,
    float reward
) {
    int2 hiddenColumnPosition = (int2)(get_global_id(0), get_global_id(1));
	
    int hiddenColumnIndex = address2(hiddenColumnPosition, hiddenSize.xy);

    float rescale = 1.0f / hiddenCounts[hiddenColumnIndex];

    // Max value
    float nextQ = hiddenActivations[address3((int3)(hiddenColumnPosition, 0), hiddenSize)];
    
    // Find max
    for (int c = 1; c < hiddenSize.z; c++)
        nextQ = fmax(nextQ, hiddenActivations[address3((int3)(hiddenColumnPosition, c), hiddenSize)]);

    int hiddenIndexPrev = address3((int3)(hiddenColumnPosition, hiddenCsPrev[hiddenColumnIndex]), hiddenSize);

    float deltaQ = reward + gamma * nextQ * rescale - hiddenActivationsPrev[hiddenIndexPrev] * rescale;
    
    deltaOHVs(nonZeroValues, rowRanges, columnIndices, visibleCsPrev, alpha * deltaQ, hiddenIndexPrev, visibleSize.z);
}

// ------------------------------------------- Image Encoder -------------------------------------------

void kernel imForward(
    global const float* visibleActivations,
    global float* hiddenActivations,
    global float* nonZeroValues,
    global const int* rowRanges,
    global const int* columnIndices,
    int3 hiddenSize
) {
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int hiddenIndex = address3(hiddenPosition, hiddenSize);

    hiddenActivations[hiddenIndex] += multiply(nonZeroValues, rowRanges, columnIndices, visibleActivations, hiddenIndex);
}

void kernel imInhibit(
    global const float* hiddenActivations,
    global int* hiddenCs,
    int3 hiddenSize
) {
    int2 hiddenColumnPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = -999999.0f;
    
    // Find max
    for (int c = 0; c < hiddenSize.z; c++) {
        int hiddenIndex = address3((int3)(hiddenColumnPosition, c), hiddenSize);

        float value = hiddenActivations[hiddenIndex];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenColumnPosition, hiddenSize.xy)] = maxIndex;
}

void kernel imLearn(
    global const float* visibleActivations,
    global const int* hiddenCs,
    global float* nonZeroValues,
    global const int* nonZeroValueIndices,
    global const int* columnRanges,
    global const int* rowIndices,
    int3 visibleSize,
    int3 hiddenSize,
    float alpha
) {
    int3 visiblePosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int visibleIndex = address3(visiblePosition, visibleSize);

    float sum = multiplyOHVsT(nonZeroValues, columnRanges, rowIndices, nonZeroValueIndices, hiddenCs, visibleIndex, hiddenSize.z);

    sum /= (countsT(columnRanges, address2(visiblePosition.xy, visibleSize.xy) * visibleSize.z) / hiddenSize.z);

    float delta = alpha * (visibleActivations[visibleIndex] - sum);

    deltaOHVsT(nonZeroValues, columnRanges, rowIndices, nonZeroValueIndices, hiddenCs, delta, visibleIndex, hiddenSize.z);
}