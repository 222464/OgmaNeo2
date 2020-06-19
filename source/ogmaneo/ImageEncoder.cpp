// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

#include <algorithm>

using namespace ogmaneo;

bool pairfiCompare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first; // Backwards so largest is in front
}

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum -= vl.weights.distance2(*inputActs[vli], hiddenIndex);
            count += vl.weights.count(hiddenIndex);
        }

        sum /= std::max(1, count);

        hiddenActivations[hiddenIndex] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void ImageEncoder::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        float sum = vl.weights.multiplyOHVsT(*hiddenCs, visibleIndex, hiddenSize.z) / std::max(1, vl.weights.countT(visibleIndex) / hiddenSize.z);

        vl.reconstructions[visibleIndex] = sum;
    }
}

void ImageEncoder::backwardErrors(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    const FloatBuffer* inputActs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        float sum = vl.weights.multiplyOHVsT(*hiddenCs, visibleIndex, hiddenSize.z) / (vl.weights.countT(visibleIndex) / hiddenSize.z);

        vl.errors[visibleIndex] = (*inputActs)[visibleIndex] - sum;
    }
}

void ImageEncoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    std::vector<std::pair<float, int>> activations(hiddenSize.z);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        activations[hc] = std::make_pair(hiddenActivations[hiddenIndex], hc);
    }

    std::sort(activations.begin(), activations.end(), pairfiCompare);

    for (int i = 0; i < hiddenSize.z; i++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, activations[i].second), hiddenSize);

        float strength = std::exp(-i * i * gamma / std::max(0.0001f, hiddenResources[hiddenIndex])) * hiddenResources[hiddenIndex];

        hiddenResources[hiddenIndex] -= alpha * strength;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.weights.hebb(*inputActs[vli], hiddenIndex, strength);
        }
    }

    int hiddenIndexMax = address3(Int3(pos.x, pos.y, hiddenCs[hiddenColumnIndex]), hiddenSize);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.weights.deltas(vl.errors, beta * hiddenResources[hiddenIndexMax], hiddenIndexMax);
    }
}

void ImageEncoder::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.weights);

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);

        // Generate transpose (needed for reconstruction)
        vl.weights.initT();

        vl.errors = FloatBuffer(numVisible, 0.0f);
        vl.reconstructions = FloatBuffer(numVisible, 0.0f);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    hiddenResources = FloatBuffer(numHidden, 1.0f);
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActs,
    bool learnEnabled
) {
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

    if (learnEnabled) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            runKernel2(cs, std::bind(ImageEncoder::backwardErrorsKernel, std::placeholders::_1, std::placeholders::_2, this, &hiddenCs, inputActs[vli], vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2);
        }

        runKernel2(cs, std::bind(ImageEncoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputActs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
    }
}

void ImageEncoder::reconstruct(
    ComputeSystem &cs,
    const IntBuffer* hiddenCs
) {
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        runKernel2(cs, std::bind(ImageEncoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenCs, vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2);
    }
}

void ImageEncoder::writeToStream(
    std::ostream &os
) const {
    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));

    writeBufferToStream(os, &hiddenCs);
    writeBufferToStream(os, &hiddenActivations);
    writeBufferToStream(os, &hiddenResources);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));

    readBufferFromStream(is, &hiddenCs);
    readBufferFromStream(is, &hiddenActivations);
    readBufferFromStream(is, &hiddenResources);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        readSMFromStream(is, vl.weights);

        vl.errors = FloatBuffer(numVisible, 0.0f);
        vl.reconstructions = FloatBuffer(numVisible, 0.0f);
    }
}