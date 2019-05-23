// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Pather.h"

#include <unordered_set>

using namespace ogmaneo;

// Planner
void ogmaneo::iterate(
    int numStates,
    int transitionsStart,
    int rewardsStart,
    const FloatBuffer &transitions,
    const FloatBuffer &rewards,
    FloatBuffer &qs,
    float gamma
) {
    for (int s = 0; s < numStates; s++) {
        for (int sp = 0; sp < numStates; sp++) {
            float maxQ = -999999.0f;

            for (int spp = 0; spp < numStates; spp++)
                maxQ = std::max(maxQ, qs[transitionsStart + sp * numStates + spp]);

            qs[transitionsStart + s * numStates + sp] = transitions[transitionsStart + s * numStates + sp] * (rewards[rewardsStart + sp] + gamma * maxQ);
        }
    }
}

int ogmaneo::getPolicy(
    int startIndex,
    int numStates,
    int transitionsStart,
    const FloatBuffer &qs
) {
    float maxQ = -999999.0f;
    int maxIndex = 0;

    for (int ap = 0; ap < numStates; ap++) {
        float q = qs[transitionsStart + startIndex * numStates + ap];
        
        if (q > maxQ) {
            maxQ = q;

            maxIndex = ap;
        }
    }

    return maxIndex;
}

void Pather::forward(
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

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._ffWeights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld._size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCsPrev[hiddenColumnIndex] = _hiddenCs[hiddenColumnIndex];
    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Pather::learnFF(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float target = (vc == targetC ? 1.0f : 0.0f);

        float sum = vl._ffWeights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._ffWeights.countsT(visibleIndex) / _hiddenSize.z);

        float delta = _ffLearnRate * (target - sum);

        vl._ffWeights.deltaOHVsT(_hiddenCs, delta, visibleIndex, _hiddenSize.z);
    }
}

void Pather::learnFB(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli,
    float reward
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    int visibleIndex = address3(Int3(pos.x, pos.y, targetC), vld._size);

    float sum = vl._fbWeights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._fbWeights.countsT(visibleIndex) / _hiddenSize.z);

    float delta = _fbLearnRate * (reward - sum);

    vl._fbWeights.deltaOHVsT(_hiddenCs, delta, visibleIndex, _hiddenSize.z);
}

void Pather::transition(
    const Int2 &pos,
    std::mt19937 &rng,
    const FloatBuffer* feedBackRewards,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int transitionsStart = hiddenColumnIndex * _hiddenSize.z * _hiddenSize.z;
    int rewardsStart = hiddenColumnIndex * _hiddenSize.z;

    if (learnEnabled) {
        int startIndex = _hiddenCsPrev[hiddenColumnIndex];
        int endIndex = _hiddenCs[hiddenColumnIndex];

        // Decay
        // for (int i = 0; i < _hiddenSize.z * _hiddenSize.z; i++) {
        //     int wi = i + weightsStart;

        //     _transitionWeights[wi] *= _beta;
        // }

        // int wi = endIndex + startIndex * _hiddenSize.z + weightsStart;

        // _transitionWeights[wi] = 1.0f;

        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            float target = (hc == endIndex ? 1.0f : 0.0f);

            int wi = hc + startIndex * _hiddenSize.z + transitionsStart;

            _transitionWeights[wi] += _tLearnRate * (target - _transitionWeights[wi]);
        }
    }

    // Iteration
    for (int it = 0; it < _iterations; it++)
        iterate(_hiddenSize.z, transitionsStart, rewardsStart, _transitionWeights, *feedBackRewards, _qs, _gamma);

    _predictedCs[hiddenColumnIndex] = getPolicy(_hiddenCs[hiddenColumnIndex], _hiddenSize.z, transitionsStart, _qs);
}

void Pather::backwardActions(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));
    
    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float sum = vl._ffWeights.multiplyOHVsT(*hiddenCs, visibleIndex, _hiddenSize.z);

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = vc;
        }
    }

    vl._visibleActions[visibleColumnIndex] = maxIndex;
}

void Pather::backwardRewards(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        vl._visibleRewards[visibleIndex] = vl._fbWeights.multiplyOHVsT(*hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._fbWeights.countsT(visibleIndex) / _hiddenSize.z);
    }
}

void Pather::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    bool isFirstLayer
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    std::uniform_real_distribution<float> ffWeightDist(0.99f, 1.0f);
    std::uniform_real_distribution<float> fbWeightDist(-0.01f, 0.0f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._ffWeights);

        // Generate transpose (needed for reconstruction)
        vl._ffWeights.initT();

        for (int i = 0; i < vl._ffWeights._nonZeroValues.size(); i++)
            vl._ffWeights._nonZeroValues[i] = ffWeightDist(cs._rng);

        if (isFirstLayer)
            vl._visibleActions = IntBuffer(numVisibleColumns, 0);
        else {
            vl._visibleRewards = FloatBuffer(numVisible, 0.0f);

            initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._fbWeights);

            vl._fbWeights.initT();

            for (int i = 0; i < vl._fbWeights._nonZeroValues.size(); i++)
                vl._fbWeights._nonZeroValues[i] = fbWeightDist(cs._rng);
        }
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
    _hiddenCsPrev = IntBuffer(numHiddenColumns, 0);

    _predictedCs = IntBuffer(numHiddenColumns, 0);

    _transitionWeights = FloatBuffer(numHidden * _hiddenSize.z, 0.0f);
}

void Pather::stepUp(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputCs);
#else
    runKernel2(cs, std::bind(Pather::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    learnFF(Int2(x, y), cs._rng, inputCs, vli);
#else
            runKernel2(cs, std::bind(Pather::learnFFKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void Pather::stepDown(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    const FloatBuffer* feedBackRewards,
    bool isFirstLayer,
    float reward,
    bool learnEnabled
) {
    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    learnFB(Int2(x, y), cs._rng, inputCs, vli, reward);
#else
            runKernel2(cs, std::bind(Pather::learnFBKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, vli, reward), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }

    // Find node on path to goal
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            transition(Int2(x, y), cs._rng, feedBackCs, learnEnabled);
#else
    runKernel2(cs, std::bind(Pather::transitionKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackRewards, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    if (isFirstLayer) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                   backwardActions(Int2(x, y), cs._rng, &_predictedCs, vli);
#else
            runKernel2(cs, std::bind(Pather::backwardActionsKernel, std::placeholders::_1, std::placeholders::_2, this, &_predictedCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
    else {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                   backwardRewards(Int2(x, y), cs._rng, &_predictedCs, vli);
#else
            runKernel2(cs, std::bind(Pather::backwardRewardsKernel, std::placeholders::_1, std::placeholders::_2, this, &_predictedCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void Pather::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_ffLearnRate), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_fbLearnRate), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_tLearnRate), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_iterations), sizeof(int));

    writeBufferToStream(os, &_hiddenCs);
    writeBufferToStream(os, &_hiddenCsPrev);

    writeBufferToStream(os, &_predictedCs);

    writeBufferToStream(os, &_transitionWeights);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._ffWeights);
        writeSMToStream(os, vl._fbWeights);

        writeBufferToStream(os, &vl._visibleRewards);
        writeBufferToStream(os, &vl._visibleActions);
    }
}

void Pather::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_ffLearnRate), sizeof(float));
    is.read(reinterpret_cast<char*>(&_fbLearnRate), sizeof(float));
    is.read(reinterpret_cast<char*>(&_tLearnRate), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&_iterations), sizeof(int));

    readBufferFromStream(is, &_hiddenCs);
    readBufferFromStream(is, &_hiddenCsPrev);

    readBufferFromStream(is, &_predictedCs);

    readBufferFromStream(is, &_transitionWeights);

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

        readSMFromStream(is, vl._ffWeights);
        readSMFromStream(is, vl._fbWeights);

        readBufferFromStream(is, &vl._visibleRewards);
        readBufferFromStream(is, &vl._visibleActions);
    }
}