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
    const FloatBuffer* feedBackStates,
    const FloatBuffer* inputStates
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = (feedBackStates == nullptr ? 0.0f : _visibleLayer._feedBackWeights.multiply(*feedBackStates, hiddenIndex)) + _visibleLayer._inputWeights.multiply(*inputStates, hiddenIndex);
        int count = _visibleLayer._feedBackWeights.count(hiddenIndex);
    
        _hiddenStates[hiddenIndex] = std::tanh(sum * std::sqrt(1.0f / std::max(1, count)));
    }
}

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    int index
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = (_historySamples[index]->_feedBackStates.empty() ? 0.0f : _visibleLayer._feedBackWeights.multiply(_historySamples[index]->_feedBackStates, hiddenIndex)) + _visibleLayer._inputWeights.multiply(_historySamples[index]->_inputStates, hiddenIndex);
        int count = _visibleLayer._feedBackWeights.count(hiddenIndex);

        float predState = std::tanh(sum * std::sqrt(1.0f / std::max(1, count)));

        float delta = _alpha * (_historySamples[index - 1]->_hiddenTargetStates[hiddenIndex] - predState) * (1.0f - predState * predState);

        if (!_historySamples[index]->_feedBackStates.empty())
            _visibleLayer._feedBackWeights.deltas(_historySamples[index]->_feedBackStates, delta, hiddenIndex);

        _visibleLayer._inputWeights.deltas(_historySamples[index]->_inputStates, delta, hiddenIndex);
    }
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

    int numVisibleColumns = _visibleLayerDesc._size.x * _visibleLayerDesc._size.y;
    int numVisible = numVisibleColumns * _visibleLayerDesc._size.z;

    // Create weight matrix for this visible layer and initialize randomly
    initSMLocalRF(_visibleLayerDesc._size, _hiddenSize, _visibleLayerDesc._radius, _visibleLayerDesc._dropRatio, _visibleLayer._feedBackWeights, cs._rng);

    _visibleLayer._inputWeights = _visibleLayer._feedBackWeights;

    std::normal_distribution<float> weightDist(0.0f, _visibleLayerDesc._scale);

    for (int i = 0; i < _visibleLayer._feedBackWeights._nonZeroValues.size(); i++) {
        _visibleLayer._feedBackWeights._nonZeroValues[i] = weightDist(cs._rng);
        _visibleLayer._inputWeights._nonZeroValues[i] = weightDist(cs._rng);
    }

    // Hidden
    _hiddenStates = FloatBuffer(numHidden, 0.0f);
}

void Predictor::activate(
    ComputeSystem &cs,
    const FloatBuffer* feedBackStates,
    const FloatBuffer* inputStates
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, feedBackStates, inputStates);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackStates, inputStates), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::learn(
    ComputeSystem &cs,
    const FloatBuffer* feedBackStates,
    const FloatBuffer* inputStates,
    const FloatBuffer* hiddenTargetStates
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Add history sample
    std::shared_ptr<HistorySample> sample = std::make_shared<HistorySample>();

    sample->_feedBackStates = (feedBackStates == nullptr ? FloatBuffer{} : *feedBackStates);
    sample->_inputStates = *inputStates;
    sample->_hiddenTargetStates = *hiddenTargetStates;

    _historySamples.insert(_historySamples.begin(), std::move(sample));

    if (_historySamples.size() > _maxHistorySamples)
        _historySamples.resize(_maxHistorySamples);

    if (_historySamples.size() > 1) {
        std::uniform_int_distribution<int> sampleDist(1, _historySamples.size() - 1);

        for (int it = 0; it < _historyIters; it++) {
            int index = sampleDist(cs._rng);

            // Learn kernel
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _hiddenSize.x; x++)
                for (int y = 0; y < _hiddenSize.y; y++)
                    learn(Int2(x, y), cs._rng, index);
#else
            runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, index), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
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

    writeBufferToStream(os, &_hiddenStates);

    os.write(reinterpret_cast<const char*>(&_visibleLayerDesc), sizeof(VisibleLayerDesc));

    writeSMToStream(os, _visibleLayer._feedBackWeights);
    writeSMToStream(os, _visibleLayer._inputWeights);
}

void Predictor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenStates);

    is.read(reinterpret_cast<char*>(&_visibleLayerDesc), sizeof(VisibleLayerDesc));

    readSMFromStream(is, _visibleLayer._feedBackWeights);
    readSMFromStream(is, _visibleLayer._inputWeights);
}