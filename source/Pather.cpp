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
int ogmaneo::findNextIndex(
    int startIndex,
    int endIndex,
    int size,
    int weightsStart,
    const FloatBuffer &weights
) {
    std::vector<float> dist(size, 0.0f);
    std::vector<int> prev(size, -1);

    std::unordered_set<int> q;

    for (int v = 0; v < size; v++)
        q.insert(v);

    dist[startIndex] = 1.0f;

    while (!q.empty()) {
        std::unordered_set<int>::iterator cit = q.begin();

        int u = *cit;
        float maxDist = dist[u];
        
        cit++;

        for (; cit != q.end(); cit++) {
            if (dist[*cit] > maxDist) {
                maxDist = dist[*cit];
                u = *cit;
            }
        }

        if (u == endIndex) {
            int prevU = u;

            while (prev[u] != -1) {
                prevU = u;
                u = prev[u];
            }

            return prevU;
        }

        q.erase(u);

        for (int n = 0; n < size; n++) {
            float w = weights[weightsStart + u * size + n];

            float alt = dist[u] * w;
            
            if (alt > dist[n]) {
                dist[n] = alt;

                prev[n] = u;
            }
        }
    }

    return startIndex;
}

void Pather::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
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

    if (learnEnabled) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, maxIndex), _hiddenSize);

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.hebbDecreasingOHVs(*inputCs[vli], hiddenIndex, vld._size.z, _alpha);
        }
    }
}

void Pather::transition(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* feedBackCs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    if (learnEnabled) {
        int startIndex = _hiddenCsPrev[hiddenColumnIndex];
        int endIndex = _hiddenCs[hiddenColumnIndex];
        int predIndexPrev = _predictedCs[hiddenColumnIndex];

        // float target = (predIndexPrev == endIndex ? 1.0f : 0.0f);

        // int wi = predIndexPrev + startIndex * _hiddenSize.z + hiddenColumnIndex * _hiddenSize.z * _hiddenSize.z;

        // _transitionWeights[wi] += _beta * (target - _transitionWeights[wi]);

        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            float target = (hc == endIndex ? 1.0f : 0.0f);

            int wi = hc + startIndex * _hiddenSize.z + hiddenColumnIndex * _hiddenSize.z * _hiddenSize.z;

            _transitionWeights[wi] += _beta * (target - _transitionWeights[wi]);
        }
    }

    // Pathfind
    _predictedCs[hiddenColumnIndex] = findNextIndex(_hiddenCs[hiddenColumnIndex], (*feedBackCs)[hiddenColumnIndex], _hiddenSize.z, hiddenColumnIndex * _hiddenSize.z * _hiddenSize.z, _transitionWeights);
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

    std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

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
            forward(Int2(x, y), cs._rng, inputCs, learnEnabled);
#else
    runKernel2(cs, std::bind(Pather::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Pather::stepDown(
    ComputeSystem &cs,
    const IntBuffer* feedBackCs,
    bool learnEnabled
) {
    // Find node on path to goal
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            transition(Int2(x, y), cs._rng, feedBackCs, learnEnabled);
#else
    runKernel2(cs, std::bind(Pather::transitionKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackCs, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
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

        readBufferFromStream(is, &vl._recons);
    }
}