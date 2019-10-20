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
void Predictor::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenTargetCs,
    const IntBuffer* feedBackCs,
    const IntBuffer* inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = _visibleLayer._weights.multiplyCombinedOHVs(*feedBackCs, *inputCs, hiddenIndex, _visibleLayerDesc._size.z);

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* feedBackCs,
    int index
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    const HistorySample* s = _historySamples[index].get();
    const HistorySample* sNext = _historySamples[index + 1].get();

    int targetC = sNext->_hiddenTargetCs[hiddenColumnIndex];

    int hiddenIndex = address3(Int3(pos.x, pos.y, targetC), _hiddenSize);

    float sum = _visibleLayer._weights.multiplyCombinedOHVs(*feedBackCs, s->_inputCs, hiddenIndex, _visibleLayerDesc._size.z);
    int count = _visibleLayer._weights.count(hiddenIndex) / (_visibleLayerDesc._size.z * _visibleLayerDesc._size.z);

    float closeness = 1.0f - (_historySamples.size() - 1 - index) / static_cast<float>(_maxHistorySize - 1);
    
    float delta = _alpha * (closeness - sum / std::max(1, count));

    _visibleLayer._weights.deltaCombinedOHVs(*feedBackCs, s->_inputCs, delta, hiddenIndex, _visibleLayerDesc._size.z);
}

void Predictor::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const VisibleLayerDesc &visibleLayerDesc
) {
    _visibleLayerDesc = visibleLayerDesc;

    _hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    // Create layer
    int numVisibleColumns = _visibleLayerDesc._size.x * _visibleLayerDesc._size.y;

    // Create weight matrix for this visible layer and initialize randomly
    initSMLocalRF(Int3(_visibleLayerDesc._size.x, _visibleLayerDesc._size.y, _visibleLayerDesc._size.z * _visibleLayerDesc._size.z), _hiddenSize, _visibleLayerDesc._radius, _visibleLayer._weights);

    for (int i = 0; i < _visibleLayer._weights._nonZeroValues.size(); i++)
        _visibleLayer._weights._nonZeroValues[i] = weightDist(cs._rng);

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
}

void Predictor::activate(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs,
    const IntBuffer* feedBackCs,
    const IntBuffer* inputCs
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, hiddenTargetCs, feedBackCs, inputCs);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenTargetCs, feedBackCs, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::learn(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs,
    const IntBuffer* inputCs
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Add sample
    if (_historySamples.size() > _maxHistorySize)
        _historySamples.resize(_maxHistorySize);

    if (_historySamples.size() == _maxHistorySize) {
        // Shift
        std::shared_ptr<HistorySample> first = _historySamples.front();

        for (int t = 0; t < _maxHistorySize - 1; t++) {
            _historySamples[t] = _historySamples[t + 1];
        }

        first->_inputCs = *inputCs;
        first->_hiddenTargetCs = *hiddenTargetCs;

        _historySamples.back() = first;
    }
    else {
        std::shared_ptr<HistorySample> sample = std::make_shared<HistorySample>();

        sample->_inputCs = *inputCs;
        sample->_hiddenTargetCs = *hiddenTargetCs;
        
        _historySamples.push_back(sample);
    }

    if (_historySamples.size() > 1) {
        for (int t = 0; t < _historySamples.size() - 1; t++) {
            // Learn kernel
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _hiddenSize.x; x++)
                for (int y = 0; y < _hiddenSize.y; y++)
                    learn(Int2(x, y), cs._rng, inputCs, t);
#else
            runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, t), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void Predictor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);

    os.write(reinterpret_cast<const char*>(&_visibleLayerDesc), sizeof(VisibleLayerDesc));

    writeSMToStream(os, _visibleLayer._weights);
}

void Predictor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

    is.read(reinterpret_cast<char*>(&_visibleLayerDesc), sizeof(VisibleLayerDesc));

    readSMFromStream(is, _visibleLayer._weights);
}