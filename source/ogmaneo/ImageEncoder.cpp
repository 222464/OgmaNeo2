// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActivations
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // Find center
    int hiddenIndex0 = address3(Int3(pos.x, pos.y, 0), _hiddenSize);

    float center = 0.0f;
    int count = 0;

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        center += vl._weights.total(*inputActivations[vli], hiddenIndex0);
        count += vl._weights.count(hiddenIndex0);
    }

    center /= std::max(1, count);

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiplyBiased(*inputActivations[vli], hiddenIndex, -center);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
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

    std::normal_distribution<float> weightDist(0.0f, 1.0f);

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
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActivations
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActivations);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void ImageEncoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    writeBufferToStream(os, &_hiddenCs);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    readBufferFromStream(is, &_hiddenCs);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl._weights);
    }
}