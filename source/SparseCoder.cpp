// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"
#include <iostream>
using namespace ogmaneo;

void SparseCoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld._size.z);
            count += vl._weights.count(hiddenIndex) / vld._size.z;
        }

        sum /= std::max(1, count);

        _hiddenActivations[hiddenIndex] = _hiddenStimuli[hiddenIndex] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::inhibit(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = _laterals.multiplyNoDiagonalOHVs(_hiddenCsTemp, hiddenIndex, _hiddenSize.z) / std::max(1, _laterals.count(hiddenIndex) / _hiddenSize.z - 1); // -1 for missing diagonal
        
        _hiddenActivations[hiddenIndex] += _hiddenStimuli[hiddenIndex] - sum;

        if (_hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = _hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int hiddenIndexMax = address3(Int3(pos.x, pos.y, _hiddenCs[hiddenColumnIndex]), _hiddenSize);

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._weights.hebbOHVs(*inputCs[vli], hiddenIndexMax, vld._size.z, 1.0f / (1.0f + _hiddenUsages[hiddenIndexMax]));
    }

    _laterals.hebbOHVs(_hiddenCs, hiddenIndexMax, _hiddenSize.z, 1.0f / (1.0f + _hiddenUsages[hiddenIndexMax]));

    _hiddenUsages[hiddenIndexMax] = std::min(999999.0f, _hiddenUsages[hiddenIndexMax] + 1.0f);
}

void SparseCoder::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    int lateralRadius,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;
    _lateralRadius = lateralRadius;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    std::uniform_real_distribution<float> forwardWeightDist(0.99f, 1.0f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = forwardWeightDist(cs._rng);
    }

    _hiddenStimuli = FloatBuffer(numHidden, 0.0f);
    _hiddenActivations = FloatBuffer(numHidden, 0.0f);

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
    _hiddenCsTemp = IntBuffer(numHiddenColumns);

    _hiddenUsages = FloatBuffer(numHidden, 0.0f);

    std::uniform_real_distribution<float> lateralWeightDist(0.0f, 0.01f);

    initSMLocalRF(_hiddenSize, _hiddenSize, _lateralRadius, _laterals);

    for (int i = 0; i < _laterals._nonZeroValues.size(); i++)
        _laterals._nonZeroValues[i] = lateralWeightDist(cs._rng);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputCs);
#else
    runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    // Iterate
    for (int it = 0; it < _explainIters; it++) {
        // Update temps
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < numHiddenColumns; x++)
            copyInt(x, cs._rng, &_hiddenCs, &_hiddenCsTemp);
#else
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, &_hiddenCsTemp), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                inhibit(Int2(x, y), cs._rng);
#else
        runKernel2(cs, std::bind(SparseCoder::inhibitKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
    }

    if (learnEnabled) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                learn(Int2(x, y), cs._rng, inputCs);
#else
        runKernel2(cs, std::bind(SparseCoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
    }
}

void SparseCoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_explainIters), sizeof(int));

    writeBufferToStream(os, &_hiddenCs);
    writeBufferToStream(os, &_hiddenUsages);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);
    }
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_explainIters), sizeof(int));

    readBufferFromStream(is, &_hiddenCs);
    readBufferFromStream(is, &_hiddenUsages);

    _hiddenStimuli = FloatBuffer(numHidden, 0.0f);
    _hiddenActivations = FloatBuffer(numHidden, 0.0f);

    _hiddenCsTemp = IntBuffer(numHiddenColumns);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        readSMFromStream(is, vl._weights);
    }
}