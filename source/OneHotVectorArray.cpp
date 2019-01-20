// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "OneHotVectorArray.h"

OneHotVectorArray::OneHotVectorArray(
	int vectorSize,
	int vectorCount
) :
	_vectorSize(vectorSize)
{
	_indices.resize(vectorCount);
}

void OneHotVectorArray::fromVectorMax(
	const std::vector<float> &data
) {
	for (int vectorIndex = 0; vectorIndex < _vectorSize; ++vectorIndex) {
		int vector_offset = vectorIndex * _vectorSize;
		int max_index = 0;
		float max_value = 0.0f;

		for (int i = 0; i < _vectorSize; ++i) {
			int index = vector_offset + i;

			float value = data[index];

			if (value > max_value) {
				max_index = index;
				max_value = value;
			}
		}

		_indices[vector_offset] = max_index;
	}
}