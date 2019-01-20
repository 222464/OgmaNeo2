// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

struct OneHotVectorArray {
	int _vectorSize;
	std::vector<int> _indices;

	OneHotVectorArray() {}

	OneHotVectorArray(
		int vectorSize,
		int vectorCount
	);

	void fromVectorMax(
		const std::vector<float> &data
	);
};