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

// Pathfinder
std::pair<int, float> ogmaneo::findNextIndex(
    int startIndex,
    int endIndex,
    int size,
    int transitionsStart,
    const FloatBuffer &transitions,
    float gamma
) {
    std::vector<float> prob(size, 0.0f);
    std::vector<int> prev(size, -1);

    std::unordered_set<int> q;

    for (int v = 0; v < size; v++)
        q.insert(v);

    prob[startIndex] = 1.0f;

    while (!q.empty()) {
        std::unordered_set<int>::iterator cit = q.begin();

        int u = *cit;
        float maxProb = prob[u];
        
        cit++;

        for (; cit != q.end(); cit++) {
            if (prob[*cit] > maxProb) {
                maxProb = prob[*cit];
                u = *cit;
            }
        }

        if (u == endIndex) {
            int prevU = u;

            while (prev[u] != -1) {
                prevU = u;
                u = prev[u];
            }

            return std::make_pair(prevU, maxProb);
        }

        q.erase(u);

        cit = q.begin();

        for (; cit != q.end(); cit++) {
            float w = transitions[transitionsStart + u * size + *cit];

            float alt = prob[u] * w * gamma;
            
            if (alt > prob[*cit]) {
                prob[*cit] = alt;

                prev[*cit] = u;
            }
        }
    }

    return std::make_pair(startIndex, 0.0f);
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

            sum += vl._weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld._size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCsPrev[hiddenColumnIndex] = _hiddenCs[hiddenColumnIndex];
    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Pather::learnWeights(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    std::vector<float> recons(vld._size.z);
    int maxIndex = 0;

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        recons[vc] = vl._weights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._visibleCounts[visibleColumnIndex]);

        if (recons[vc] > recons[maxIndex])
            maxIndex = vc;
    }

    if (maxIndex != targetC) {
        for (int vc = 0; vc < vld._size.z; vc++) {
            int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

            float target = (vc == targetC ? 1.0f : 0.0f);
                
            float delta = _alpha * (target - recons[vc]);//(recons[vc] > 0.0f ? 1.0f + recons[vc] : std::exp(recons[vc]))

            vl._weights.deltaOHVsT(_hiddenCs, delta, visibleIndex, _hiddenSize.z);
        }
    }
}

void Pather::transition(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* feedBackGoalCs,
    const IntBuffer* localGoalCs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int weightsStart = hiddenColumnIndex * _hiddenSize.z * _hiddenSize.z;

    if (learnEnabled) {
        int startIndex = _hiddenCsPrev[hiddenColumnIndex];
        int endIndex = _hiddenCs[hiddenColumnIndex];
        int predIndexPrev = _predictedCs[hiddenColumnIndex];

        // Decay
        // for (int i = 0; i < _hiddenSize.z * _hiddenSize.z; i++) {
        //     int wi = i + weightsStart;

        //     _transitionWeights[wi] *= _beta;
        // }

        // int wi = endIndex + startIndex * _hiddenSize.z + weightsStart;

        // _transitionWeights[wi] = 1.0f;

        // for (int hc = 0; hc < _hiddenSize.z; hc++) {
        //     float target = (hc == endIndex ? 1.0f : 0.0f);

        //     int wi = hc + startIndex * _hiddenSize.z + weightsStart;

        //     _transitionWeights[wi] += _beta * (target - _transitionWeights[wi]);
        // }

        if (predIndexPrev != endIndex) {
            int wi = predIndexPrev + startIndex * _hiddenSize.z + weightsStart;

            _transitionWeights[wi] += _beta * (0.0f - _transitionWeights[wi]);
        }

        int wi = endIndex + startIndex * _hiddenSize.z + weightsStart;

        _transitionWeights[wi] += _beta * (1.0f - _transitionWeights[wi]);
    }

    // Pathfind
    std::pair<int, float> feedBackPath = findNextIndex(_hiddenCs[hiddenColumnIndex], (*feedBackGoalCs)[hiddenColumnIndex], _hiddenSize.z, weightsStart, _transitionWeights, _gamma);
    std::pair<int, float> localPath = findNextIndex(_hiddenCs[hiddenColumnIndex], (*localGoalCs)[hiddenColumnIndex], _hiddenSize.z, weightsStart, _transitionWeights, _gamma);

    if (feedBackPath.second > localPath.second)
        _predictedCs[hiddenColumnIndex] = feedBackPath.first;
    else
        _predictedCs[hiddenColumnIndex] = localPath.first;
}

void Pather::reconstruct(
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

        float sum = vl._weights.multiplyOHVsT(*hiddenCs, visibleIndex, _hiddenSize.z);

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = vc;
        }
    }

    vl._recons[visibleColumnIndex] = maxIndex;
}

void Pather::initRandom(
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

    std::uniform_real_distribution<float> weightDist(0.99f, 1.0f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        // Generate transpose (needed for reconstruction)
        vl._weights.initT();

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = weightDist(cs._rng);

        // Counts
        vl._visibleCounts = IntBuffer(numVisibleColumns);

        for (int i = 0; i < numVisibleColumns; i++)
            vl._visibleCounts[i] = vl._weights.countsT(i * vld._size.z) / _hiddenSize.z;

        vl._recons = IntBuffer(numVisibleColumns, 0);
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
                    learnWeights(Int2(x, y), cs._rng, inputCs, vli);
#else
            runKernel2(cs, std::bind(Pather::learnWeightsKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void Pather::stepDown(
    ComputeSystem &cs,
    const IntBuffer* feedBackGoalCs,
    const IntBuffer* localGoalCs,
    bool learnEnabled
) {
    // Find node on path to goal
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            transition(Int2(x, y), cs._rng, feedBackGoalCs, localGoalCs, learnEnabled);
#else
    runKernel2(cs, std::bind(Pather::transitionKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackGoalCs, localGoalCs, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                reconstruct(Int2(x, y), cs._rng, &_predictedCs, vli);
#else
        runKernel2(cs, std::bind(Pather::reconstructKernel, std::placeholders::_1, std::placeholders::_2, this, &_predictedCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
    }
}

void Pather::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(float));

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

        writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._visibleCounts);

        writeBufferToStream(os, &vl._recons);
    }
}

void Pather::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));

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

        readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._visibleCounts);

        readBufferFromStream(is, &vl._recons);
    }
}