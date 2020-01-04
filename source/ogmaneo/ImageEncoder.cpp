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
    const std::vector<const FloatBuffer*> &inputActs
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

            sum += vl._encWeights.multiply(*inputActs[vli], hiddenIndex);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void ImageEncoder::backward(
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

        float sum = vl._decWeights.multiplyOHVs(*hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._decWeights.count(visibleIndex) / _hiddenSize.z);

        vl._reconActs[visibleIndex] = sum;
    }
}

void ImageEncoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const FloatBuffer* inputActs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    float totalError2 = 0.0f;
    std::vector<float> activations(vld._size.z);

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float sum = vl._decWeights.multiplyOHVs(_hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._decWeights.count(visibleIndex) / _hiddenSize.z);

        activations[vc] = sum;

        float err = (*inputActs)[visibleIndex] - sum;

        totalError2 += err * err;
    }

    if ((totalError2 / vld._size.z) > _epsilon) {
        for (int vc = 0; vc < vld._size.z; vc++) {
            int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

            float delta = _alpha * ((*inputActs)[visibleIndex] - activations[vc]);

            vl._decWeights.deltaOHVs(_hiddenCs, delta, visibleIndex, _hiddenSize.z);
        }
    }
}

void ImageEncoder::initRandom(
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

    std::normal_distribution<float> encWeightDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> decWeightDist(0.0f, 1.0f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._encRadius, vl._encWeights);
        initSMLocalRF(_hiddenSize, vld._size, vld._decRadius, vl._decWeights);

        for (int i = 0; i < vl._encWeights._nonZeroValues.size(); i++)
            vl._encWeights._nonZeroValues[i] = encWeightDist(cs._rng);

        for (int i = 0; i < vl._decWeights._nonZeroValues.size(); i++)
            vl._decWeights._nonZeroValues[i] = decWeightDist(cs._rng);

        vl._reconActs = FloatBuffer(numVisible, 0.0f);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
}

void ImageEncoder::activate(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActs
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActs);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void ImageEncoder::learn(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &targetActs
) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                learn(Int2(x, y), cs._rng, targetActs[vli], vli);
#else
        runKernel2(cs, std::bind(ImageEncoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, targetActs[vli], vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
    }
}

void ImageEncoder::reconstruct(
    ComputeSystem &cs,
    const IntBuffer* hiddenCs
) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                backward(Int2(x, y), cs._rng, hiddenCs, vli);
#else
        runKernel2(cs, std::bind(ImageEncoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
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
    os.write(reinterpret_cast<const char*>(&_epsilon), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._encWeights);
        writeSMToStream(os, vl._decWeights);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_epsilon), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

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

        readSMFromStream(is, vl._encWeights);
        readSMFromStream(is, vl._decWeights);

        vl._reconActs = FloatBuffer(numVisible, 0.0f);
    }
}