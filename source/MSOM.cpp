// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "MSOM.h"

using namespace ogmaneo;

void MSOM::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputs
) {
    int hiddenIndex = address2C(pos, _hiddenSize);

    float sum = 0.0f;

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        sum -= vl._weights.distance(*inputs[vli], hiddenIndex);
    }

    _hiddenActivations[hiddenIndex] = sum;
}

void MSOM::inhibit(
    const Int2 &pos,
    std::mt19937 &rng,
    FloatBuffer* hiddenStates
) {
    int hiddenIndex = address2C(pos, _hiddenSize);

    bool highest = true;

    for (int dx = -_inhibitRadius; dx <= _inhibitRadius; dx++)
        for (int dy = -_inhibitRadius; dy <= _inhibitRadius; dy++) {
            if (dx == 0 && dy == 0)
                continue;

            Int2 dPos(pos.x + dx, pos.y + dy);

            if (inBounds0(dPos, _hiddenSize) && _hiddenActivations[address2C(dPos, _hiddenSize)] > _hiddenActivations[hiddenIndex]) {
                highest = false;

                break;
            }
        }

    (*hiddenStates)[hiddenIndex] = highest ? 1.0f : 0.0f;
}

void MSOM::blur(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenIndex = address2C(pos, _hiddenSize);

    float m = 0.0f;

    for (int dx = -_blurRadius; dx <= _blurRadius; dx++)
        for (int dy = -_blurRadius; dy <= _blurRadius; dy++) {
            Int2 dPos(pos.x + dx, pos.y + dy);

            if (inBounds0(dPos, _hiddenSize) && _hiddenStates[address2C(dPos, _hiddenSize)] != 0.0f) {
                float falloff = 1.0f - (std::abs(dx) + std::abs(dy)) / (2.0f * _blurRadius);

                m = std::max(m, falloff);
            }
        }

    _hiddenBlurs[hiddenIndex] = m;
}

void MSOM::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const FloatBuffer* hiddenStates,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleIndex = address2C(pos, vld._size);

    vl._recons[visibleIndex] = vl._weights.multiplyT(*hiddenStates, visibleIndex) / std::max(0.0001f, vl._weights.countsT(*hiddenStates, visibleIndex));
}

void MSOM::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputs
) {
    int hiddenIndex = address2C(pos, _hiddenSize);

    if (_hiddenBlurs[hiddenIndex] != 0.0f) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.hebb(*inputs[vli], hiddenIndex, _alpha * _hiddenBlurs[hiddenIndex]);
        }

        _crossWeights.hebb(_hiddenStatesPrev, hiddenIndex, _beta * _hiddenBlurs[hiddenIndex]);
    
        if (!_feedBackStatesPrev.empty()) {
            assert(!_feedBackWeights._nonZeroValues.empty());

            _feedBackWeights.hebb(_feedBackStatesPrev, hiddenIndex, _beta * _hiddenBlurs[hiddenIndex]);
        }
    }
}

void MSOM::predict(
    const Int2 &pos,
    std::mt19937 &rng,
    const FloatBuffer* feedBackStates
) {
    int hiddenIndex = address2C(pos, _hiddenSize);

    _hiddenActivations[hiddenIndex] = -_crossWeights.distance(_hiddenStates, hiddenIndex);

    if (feedBackStates != nullptr) {
        assert(!_feedBackWeights._nonZeroValues.empty());

        _hiddenActivations[hiddenIndex] += -_feedBackWeights.distance(*feedBackStates, hiddenIndex);
    }
}

void MSOM::initRandom(
    ComputeSystem &cs,
    const Int2 &hiddenSize,
    int predRadius,
    bool hasFeedBack,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _predRadius = predRadius;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHidden = _hiddenSize.x * _hiddenSize.y;

    std::uniform_real_distribution<float> weightDist(0.0f, 0.0001f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisible = vld._size.x * vld._size.y;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = weightDist(cs._rng);

        vl._weights.initT();

        vl._recons = FloatBuffer(numVisible, 0.0f);
    }

    // Hidden
    _hiddenActivations = FloatBuffer(numHidden, 0.0f);
    _hiddenStates = FloatBuffer(numHidden, 0.0f);
    _hiddenBlurs = FloatBuffer(numHidden, 0.0f);
    _hiddenPredictions = FloatBuffer(numHidden, 0.0f);

    _hiddenStatesPrev = FloatBuffer(numHidden, 0.0f);

    initSMLocalRF(_hiddenSize, _hiddenSize, _predRadius, _crossWeights);

    if (hasFeedBack) {
        _feedBackStatesPrev = FloatBuffer(numHidden, 0.0f);

        initSMLocalRF(_hiddenSize, _hiddenSize, _predRadius, _feedBackWeights);
    }
}

void MSOM::activate(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputs
) {
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputs);
#else
    runKernel2(cs, std::bind(MSOM::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            inhibit(Int2(x, y), cs._rng, &_hiddenStates);
#else
    runKernel2(cs, std::bind(MSOM::inhibitKernel, std::placeholders::_1, std::placeholders::_2, this, &_hiddenStates), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            blur(Int2(x, y), cs._rng);
#else
    runKernel2(cs, std::bind(MSOM::blurKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void MSOM::learn(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputs
) {
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            learn(Int2(x, y), cs._rng, inputs);
#else
    runKernel2(cs, std::bind(MSOM::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void MSOM::reconstruct(
    ComputeSystem &cs,
    const FloatBuffer* hiddenStates
) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                backward(Int2(x, y), cs._rng, hiddenStates, vli);
#else
        runKernel2(cs, std::bind(MSOM::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenStates, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
    }
}

void MSOM::predict(
    ComputeSystem &cs,
    const FloatBuffer* feedBackStates
) {
    int numHidden = _hiddenSize.x * _hiddenSize.y;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            predict(Int2(x, y), cs._rng, feedBackStates);
#else
    runKernel2(cs, std::bind(MSOM::predictKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackStates), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            inhibit(Int2(x, y), cs._rng, &_hiddenPredictions);
#else
    runKernel2(cs, std::bind(MSOM::inhibitKernel, std::placeholders::_1, std::placeholders::_2, this, &_hiddenPredictions), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < numHidden; x++)
        copyFloat(x, cs._rng, &_hiddenStates, &_hiddenStatesPrev);
#else
    runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenStates, &_hiddenStatesPrev), numHidden, cs._rng, cs._batchSize1);
#endif

    if (!_feedBackStatesPrev.empty()) {
        assert(feedBackStates != nullptr);

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < numHidden; x++)
            copyFloat(x, cs._rng, feedBackStates, &_feedBackStatesPrev);
#else
        runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, feedBackStates, &_feedBackStatesPrev), numHidden, cs._rng, cs._batchSize1);
#endif
    }
}

void MSOM::writeToStream(
    std::ostream &os
) const {
    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_inhibitRadius), sizeof(int));
    os.write(reinterpret_cast<const char*>(&_blurRadius), sizeof(int));

    writeBufferToStream(os, &_hiddenActivations);
    writeBufferToStream(os, &_hiddenStates);
    writeBufferToStream(os, &_hiddenBlurs);
    writeBufferToStream(os, &_hiddenPredictions);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._recons);
    }
}

void MSOM::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_inhibitRadius), sizeof(int));
    is.read(reinterpret_cast<char*>(&_blurRadius), sizeof(int));

    readBufferFromStream(is, &_hiddenActivations);
    readBufferFromStream(is, &_hiddenStates);
    readBufferFromStream(is, &_hiddenBlurs);
    readBufferFromStream(is, &_hiddenPredictions);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._recons);
    }
}