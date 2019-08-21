// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::clear(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._deltas.fill(hiddenIndex, 0.0f);
            vl._sigmas.fill(hiddenIndex, 0.0f);
        }
    }
}

void SparseCoder::activate(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = 0.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum0 = 0.0f;
        float sum1 = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum0 += vl._tBU.artActivate0(*inputCs[vli], hiddenIndex, vld._size.z, vl._deltas);
            sum1 += vl._tBU.artActivate1(hiddenIndex, vld._size.z, vl._sigmas);
        }

        float act = sum0 + (1.0f - _alpha) * sum1;

        if (act > maxActivation) {
            maxActivation = act;

            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::match0(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli,
    bool isLast
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    if (isLast) {
        for (int vc = 0; vc < vld._size.z; vc++) {
            int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

            vl._f1[visibleIndex] = vl._tTD.artMatchT(_hiddenCs, visibleIndex, _hiddenSize.z);
        }
    }
    else {
        int targetC = (*inputCs[vli])[visibleColumnIndex];

        int visibleIndex = address3(Int3(pos.x, pos.y, targetC), vld._size);

        vl._f1[visibleIndex] = vl._tTD.artMatchT(_hiddenCs, visibleIndex, _hiddenSize.z);
    }

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    int visibleIndex = address3(Int3(pos.x, pos.y, targetC), vld._size);

    if (!isLast && vl._f1[visibleIndex] < _minVigilance)
        vl._deltas.artDepleteDeltaT(_hiddenCs, visibleIndex, _hiddenSize.z, vl._tBU);
}

void SparseCoder::match1(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, _hiddenCs[hiddenColumnIndex]), _hiddenSize);

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._sigmas.artDepleteSigma(*inputCs[vli], vl._f1, hiddenIndex, vld._size.z, vl._tBU, _minVigilance);
    }
}

void SparseCoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, _hiddenCs[hiddenColumnIndex]), _hiddenSize);

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

         vl._tTD.artLearnTTD(*inputCs[vli], vl._f1, hiddenIndex, vld._size.z, vl._tBU, _beta);
    }
}

void SparseCoder::initRandom(
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

    _hiddenCounts = IntBuffer(numHiddenColumns, 0);

    std::uniform_real_distribution<float> smallDist(0.0f, 0.001f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._f1 = FloatBuffer(numVisible, 0.0f);

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._tBU);
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._tTD);
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._deltas);
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._sigmas);

        for (int i = 0; i < vl._tBU._nonZeroValues.size(); i++) {
            vl._tBU._nonZeroValues[i] = smallDist(cs._rng);
            vl._tTD._nonZeroValues[i] = 0.0f;
            vl._deltas._nonZeroValues[i] = 0.0f;
            vl._sigmas._nonZeroValues[i] = 0.0f;
        }

        // Generate transposes (needed for reconstruction)
        vl._tBU.initT();
        vl._tTD.initT();
        vl._deltas.initT();
        vl._sigmas.initT();

        // Counts
        vl._visibleCounts = IntBuffer(numVisibleColumns);

        for (int i = 0; i < numVisibleColumns; i++)
            vl._visibleCounts[i] = vl._tTD.countsT(i * vld._size.z) / _hiddenSize.z;

        for (int i = 0; i < numHiddenColumns; i++)
            _hiddenCounts[i] += vl._tBU.counts(i * _hiddenSize.z) / vld._size.z;
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
    _hiddenCsPrev = IntBuffer(numHiddenColumns, 0);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Copy to prev
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < numHiddenColumns; x++)
        copyInt(x, cs._rng, &_hiddenCs, &_hiddenCsPrev);
#else
    runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, &_hiddenCsPrev), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Clear
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            clear(Int2(x, y), cs._rng);
#else
    runKernel2(cs, std::bind(SparseCoder::clearKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    for (int it = 0; it < _iters; it++) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                activate(Int2(x, y), cs._rng, inputCs);
#else
        runKernel2(cs, std::bind(SparseCoder::activateKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    match0(Int2(x, y), cs._rng, inputCs, vli, it == _iters - 1);
#else
            runKernel2(cs, std::bind(SparseCoder::match0Kernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, vli, it == _iters - 1), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }

        if (it < _iters - 1) {
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _hiddenSize.x; x++)
                for (int y = 0; y < _hiddenSize.y; y++)
                    match1(Int2(x, y), cs._rng, inputCs);
#else
            runKernel2(cs, std::bind(SparseCoder::match1Kernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
        }
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

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);
    writeBufferToStream(os, &_hiddenCsPrev);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        //writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._visibleCounts);
    }
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);
    readBufferFromStream(is, &_hiddenCsPrev);

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

        //readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._visibleCounts);
    }
}