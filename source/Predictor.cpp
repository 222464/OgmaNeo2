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
    const std::vector<const IntBuffer*> &inputCsPlus,
    const std::vector<const IntBuffer*> &inputCsMinus
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = (hc == _hiddenCs[hiddenColumnIndex] ? 1.0f : 0.0f);

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiplyOHVs(*inputCsPlus[vli], hiddenIndex, vld._size.z);
            sum -= vl._weights.multiplyOHVs(*inputCsMinus[vli], hiddenIndex, vld._size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

// void Predictor::learn(
//     const Int2 &pos,
//     std::mt19937 &rng,
//     const IntBuffer* hiddenTargetCs,
//     const std::vector<const IntBuffer*> &inputCsPlus,
//     const std::vector<const IntBuffer*> &inputCsMinus
// ) {
//     int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

//     int maxIndex = 0;
//     float maxActivation = -999999.0f;

//     int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

//     for (int hc = 0; hc < _hiddenSize.z; hc++) {
//         int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

//         float sum = 0.0f;

//         // For each visible layer
//         for (int vli = 0; vli < _visibleLayers.size(); vli++) {
//             VisibleLayer &vl = _visibleLayers[vli];
//             const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

//             sum += vl._weights.multiplyOHVs(*inputCsPlus[vli], hiddenIndex, vld._size.z);
//             sum -= vl._weights.multiplyOHVs(*inputCsMinus[vli], hiddenIndex, vld._size.z);
//         }

//         sum /= std::max(1, _hiddenCounts[hiddenColumnIndex]);

//         float target = (hc == targetC ? 1.0f : 0.0f);

//         float delta = _alpha * (target - sigmoid(sum));

//         for (int vli = 0; vli < _visibleLayers.size(); vli++) {
//             VisibleLayer &vl = _visibleLayers[vli];
//             const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

//             vl._weights.deltaOHVs(*inputCsPlus[vli], delta, hiddenIndex, vld._size.z);
//             vl._weights.deltaOHVs(*inputCsMinus[vli], -delta, hiddenIndex, vld._size.z);
//         }
//     }
// }

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenTargetCs,
    const std::vector<const IntBuffer*> &inputCsPlus,
    const std::vector<const IntBuffer*> &inputCsMinus
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = _hiddenBiases[hiddenIndex];

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiplyOHVs(*inputCsPlus[vli], hiddenIndex, vld._size.z);
            sum -= vl._weights.multiplyOHVs(*inputCsMinus[vli], hiddenIndex, vld._size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = hc;
        }
    }
    
    int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

    if (maxIndex != targetC) {
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), _hiddenSize);
        int hiddenIndexTarget = address3(Int3(pos.x, pos.y, targetC), _hiddenSize);

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.deltaOHVs(*inputCsPlus[vli], _alpha, hiddenIndexTarget, vld._size.z);
            vl._weights.deltaOHVs(*inputCsMinus[vli], -_alpha, hiddenIndexTarget, vld._size.z);

            vl._weights.deltaOHVs(*inputCsPlus[vli], -_alpha, hiddenIndexMax, vld._size.z);
            vl._weights.deltaOHVs(*inputCsMinus[vli], _alpha, hiddenIndexMax, vld._size.z);
        }

        _hiddenBiases[hiddenIndexMax] -= _alpha;
        _hiddenBiases[hiddenIndexTarget] += _alpha;
    }
}

void Predictor::initRandom(
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

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = weightDist(cs._rng);

        for (int i = 0; i < numHiddenColumns; i++)
            _hiddenCounts[i] += vl._weights.counts(i * _hiddenSize.z) / vld._size.z;
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);

    _hiddenBiases = FloatBuffer(numHidden, 0.0f);
}

void Predictor::activate(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCsPlus,
    const std::vector<const IntBuffer*> &inputCsMinus
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputCsPlus, inputCsMinus);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCsPlus, inputCsMinus), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::learn(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs,
    const std::vector<const IntBuffer*> &inputCsPlus,
    const std::vector<const IntBuffer*> &inputCsMinus
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Learn kernel
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            learn(Int2(x, y), cs._rng, hiddenTargetCs, inputCsPlus, inputCsMinus);
#else
    runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenTargetCs, inputCsPlus, inputCsMinus), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);

    writeBufferToStream(os, &_hiddenCounts);

    writeBufferToStream(os, &_hiddenBiases);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);
    }
}

void Predictor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

    readBufferFromStream(is, &_hiddenCounts);

    readBufferFromStream(is, &_hiddenBiases);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl._weights);
    }
}