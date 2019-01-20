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
		int vectorOffset = vectorIndex * _vectorSize;
		int maxIndex = 0;
		float maxValue = 0.0f;

		for (int i = 0; i < _vectorSize; ++i) {
			int index = vectorOffset + i;

			float value = data[index];

			if (value > maxValue) {
				maxIndex = index;
				maxValue = value;
			}
		}

		_indices[vectorOffset] = maxIndex;
	}
}