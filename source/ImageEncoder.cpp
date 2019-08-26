// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActivations
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiply(*inputActivations[vli], hiddenIndex);
        }

        sum /= std::max(1, _hiddenCounts[hiddenColumnIndex]);

        _hiddenActivations[hiddenIndex] = _hiddenStimuli[hiddenIndex] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCsTemp[hiddenColumnIndex] = maxIndex;
}

void ImageEncoder::inhibit(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = _laterals.multiplyNoDiagonalOHVs(_hiddenCsTemp, hiddenIndex, _hiddenSize.z) / std::max(1, _laterals.counts(hiddenIndex) / _hiddenSize.z - 1); // -1 for missing diagonal

        _hiddenActivations[hiddenIndex] += (1.0f - sum) * _hiddenStimuli[hiddenIndex];

        if (_hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = _hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void ImageEncoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActivations
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, _hiddenCs[hiddenColumnIndex]), _hiddenSize);

    if (_hiddenActivations[hiddenIndex] / (1 + _explainIters) > _minVigilance) {
        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.hebb(*inputActivations[vli], hiddenIndex, _alpha);
        }

        _laterals.hebbOHVs(_hiddenCs, hiddenIndex, _hiddenSize.z, _beta);
    }
}

void ImageEncoder::initRandom(
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

    _hiddenCounts = IntBuffer(numHiddenColumns, 0);

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

        for (int i = 0; i < numHiddenColumns; i++)
            _hiddenCounts[i] += vl._weights.counts(i * _hiddenSize.z);
    }

    _hiddenStimuli = FloatBuffer(numHidden, 0.0f);
    _hiddenActivations = FloatBuffer(numHidden, 0.0f);

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
    _hiddenCsTemp = IntBuffer(numHiddenColumns, 0);

    std::uniform_real_distribution<float> lateralWeightDist(0.0f, 0.01f);

    initSMLocalRF(_hiddenSize, _hiddenSize, _lateralRadius, _laterals);

    for (int i = 0; i < _laterals._nonZeroValues.size(); i++)
        _laterals._nonZeroValues[i] = lateralWeightDist(cs._rng);
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActivations,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActivations);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    // Iterate
    for (int it = 0; it < _explainIters; it++) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                inhibit(Int2(x, y), cs._rng);
#else
        runKernel2(cs, std::bind(ImageEncoder::inhibitKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

        if (it < _explainIters - 1) {
            // Update temps
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < numHiddenColumns; x++)
                copyInt(x, cs._rng, &_hiddenCs, &_hiddenCsTemp);
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, &_hiddenCsTemp), numHiddenColumns, cs._rng, cs._batchSize1);
#endif
        }
    }

    if (learnEnabled) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                learn(Int2(x, y), cs._rng, inputActivations);
#else
        runKernel2(cs, std::bind(ImageEncoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
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
    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_explainIters), sizeof(int));
    os.write(reinterpret_cast<const char*>(&_minVigilance), sizeof(float));

    writeBufferToStream(os, &_hiddenCounts);

    writeBufferToStream(os, &_hiddenCs);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_explainIters), sizeof(int));
    is.read(reinterpret_cast<char*>(&_minVigilance), sizeof(float));

    readBufferFromStream(is, &_hiddenCounts);

    readBufferFromStream(is, &_hiddenCs);

    _hiddenCsTemp = IntBuffer(_hiddenCs.size());

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