// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

// Kernels
void ImageEncoder::init(
    int pos,
    std::mt19937 &rng,
    int vli
) {
    // Initialize weights into uniform range
	std::uniform_real_distribution<float> weightDist(0.99f, 1.0f);

    _visibleLayers[vli]._weights._nonZeroValues[pos] = weightDist(rng);
}

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActivations,
    bool learnEnabled
) {
     // --- Clear Activations ---

    for (int hc = 0; hc < _hiddenSize.z; hc++)
        _hiddenActivations[address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y))] = 0.0f;

    // --- Multiply ---

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.multiplyRange(*inputActivations[vli], _hiddenActivations, hiddenIndex, 1);
        }
    }

    // --- Find max ---

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

        if (_hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = _hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    _hiddenCs[address2R(pos, _hiddenSize.x)] = maxIndex;

    // --- Learn ---

    int hiddenIndex = address3R(Int3(pos.x, pos.y, maxIndex), Int2(_hiddenSize.x, _hiddenSize.y));

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._weights.hebbRuleDecreasing(*inputActivations[vli], hiddenIndex, _alpha);
    }
}

void ImageEncoder::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    // Clear activations
    for (int vc = 0; vc < vld._size.z; vc++)
        vl._visibleActivations[address3R(Int3(pos.x, pos.y, vc), Int2(vld._size.x, vld._size.y))] = 0.0f;

    // --- Multiply ---
    
    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3R(Int3(pos.x, pos.y, vc), Int2(vld._size.x, vld._size.y));
    
        vl._weights.multiplyRangeOHVsT(_hiddenCs, vl._visibleActivations, visibleIndex, 1, _hiddenSize.z);
    }
}

void ImageEncoder::initRandom(
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
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < vl._weights._nonZeroValues.size(); x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(ImageEncoder::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), vl._weights._nonZeroValues.size(), cs._rng, cs._batchSize1);
#endif

        // Generate transpose (needed for reconstruction)
        vl._weights.initT();

        vl._visibleActivations = FloatBuffer(numVisible);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < numVisible; x++)
            fillFloat(x, cs._rng, &vl._visibleActivations, 0.0f);
#else
        runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &vl._visibleActivations, 0.0f), numVisible, cs._rng, cs._batchSize1);
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
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActivations,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;

#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActivations, learnEnabled);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void ImageEncoder::reconstruct(
    ComputeSystem &cs,
    const IntBuffer* hiddenCs
) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_DEBUG
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                backward(Int2(x, y), cs._rng, hiddenCs, vli);
#else
        runKernel2(cs, std::bind(ImageEncoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
    }
}

void ImageEncoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._visibleActivations);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._visibleActivations);
    }
}