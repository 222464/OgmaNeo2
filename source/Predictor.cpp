// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

// Kernels
void Predictor::init(
    int pos,
    std::mt19937 &rng,
    int vli)
 {
    // Randomly initialize weights in range
	std::uniform_real_distribution<float> weightDist(-0.0001f, 0.0001f);

    _visibleLayers[vli]._weights._nonZeroValues[pos] = weightDist(rng);
}

void Predictor::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    // --- Clear Activations ---

    for (int hc = 0; hc < _hiddenSize.z; hc++)
        _hiddenActivations[address3C(Int3(pos.x, pos.y, hc), _hiddenSize)] = 0.0f;

    // --- Multiply ---

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.multiplyRangeOHVs(*inputCs[vli], _hiddenActivations, hiddenIndex, 1, vld._size.z);
        }
    }

    // --- Find max ---

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

        if (_hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = _hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    _hiddenCs[address2C(pos, Int2(_hiddenSize.x, _hiddenSize.y))] = maxIndex;
}

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenTargetCs
) {
    int targetC = (*hiddenTargetCs)[address2C(pos, Int2(_hiddenSize.x, _hiddenSize.y))];

    // --- Find Deltas ---

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        int hiddenIndex = address3C(hiddenPosition, _hiddenSize);

        float target = (hc == targetC ? 1.0f : 0.0f);

        _hiddenDeltas[hiddenIndex] = _alpha * (target - sigmoid(_hiddenActivations[hiddenIndex])); // Delta
    }

    // --- Delta Rule ---

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.deltaRuleRangeOHVs(vl._inputCsPrev, _hiddenDeltas, hiddenIndex, 1, vld._size.z); // Apply delta rule
        }
    }
}

void Predictor::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < vl._weights._nonZeroValues.size(); x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(Predictor::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), vl._weights._nonZeroValues.size(), cs._rng, cs._batchSize1);
#endif

        vl._inputCsPrev = IntBuffer(numVisibleColumns);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < numVisibleColumns; x++)
            fillInt(x, cs._rng, &vl._inputCsPrev, 0);
#else
        runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &vl._inputCsPrev, 0), numVisibleColumns, cs._rng, cs._batchSize1);
#endif
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    _hiddenActivations = FloatBuffer(numHidden);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHidden; x++)
        fillFloat(x, cs._rng, &_hiddenActivations, 0.0f);
#else
    runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenActivations, 0.0f), numHidden, cs._rng, cs._batchSize1);
#endif

    _hiddenDeltas = FloatBuffer(numHidden);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHidden; x++)
        fillFloat(x, cs._rng, &_hiddenDeltas, 0.0f);
#else
    runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenDeltas, 0.0f), numHidden, cs._rng, cs._batchSize1);
#endif
}

void Predictor::activate(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &visibleCs
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, visibleCs);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    // Copy to prevs
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

#ifdef KERNEL_DEBUG
        for (int x = 0; x < numVisibleColumns; x++)
            copyInt(x, cs._rng, visibleCs[vli], &vl._inputCsPrev);
#else
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, visibleCs[vli], &vl._inputCsPrev), numVisibleColumns, cs._rng, cs._batchSize1);
#endif
    }
}

void Predictor::learn(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs
) {
    if (_alpha == 0.0f)
        return;

    // Learn kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            learn(Int2(x, y), cs._rng, hiddenTargetCs);
#else
    runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenTargetCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);

    writeBufferToStream(os, &_hiddenActivations);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._inputCsPrev);
    }
}

void Predictor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

    readBufferFromStream(is, &_hiddenActivations);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._inputCsPrev);
    }
}