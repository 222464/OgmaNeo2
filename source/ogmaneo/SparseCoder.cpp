// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;
    std::vector<float> activations(_hiddenSize.z);

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld._size.z);
            count += vl._weights.count(hiddenIndex) / vld._size.z;
        }

        sum /= std::max(1, count);

        activations[hc] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    float total = 0.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        activations[hc] = std::exp(activations[hc] - maxActivation);

        total += activations[hc];
    }

    int selectIndex = 0;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    float cusp = dist01(rng) * total;

    float sumSoFar = 0.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        sumSoFar += activations[hc];

        if (sumSoFar >= cusp) {
            selectIndex = hc;

            break;
        }
    }

    _hiddenRandomCs[hiddenColumnIndex] = selectIndex;

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    float maxActivation = -999999.0f;

    std::vector<float> activations(vld._size.z);

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float target = (vc == targetC ? 1.0f : 0.0f);

        float sum = vl._weights.multiplyOHVsT(_hiddenRandomCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._weights.countT(visibleIndex) / _hiddenSize.z);

        activations[vc] = sum;

        maxActivation = std::max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int vc = 0; vc < vld._size.z; vc++) {
        activations[vc] = std::exp(activations[vc] - maxActivation);

        total += activations[vc];
    }

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float delta = _alpha * ((vc == targetC ? 1.0f : 0.0f) - (activations[vc] / std::max(0.0001f, total)));

        vl._weights.deltaOHVsT(_hiddenRandomCs, delta, visibleIndex, _hiddenSize.z);
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

    std::normal_distribution<float> weightDist(0.0f, 0.1f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = weightDist(cs._rng);

        // Generate transpose (needed for reconstruction)
        vl._weights.initT();
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
    _hiddenRandomCs = IntBuffer(numHiddenColumns, 0);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputCs, learnEnabled);
#else
    runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    learn(Int2(x, y), cs._rng, inputCs, vli);
#else
            runKernel2(cs, std::bind(SparseCoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
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

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);
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
    _hiddenRandomCs = IntBuffer(numHiddenColumns, 0);

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