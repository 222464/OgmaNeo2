// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, 0), Int3(hiddenSize.x, hiddenSize.y, 1));

    float sum = 0.0f;
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        sum += vl.ffWeights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
        count += vl.ffWeights.count(hiddenIndex) / vld.size.z;
    }

    hiddenActivations[hiddenColumnIndex] = sigmoid(sum / std::max(1, count));

    hiddenCs[hiddenColumnIndex] = hiddenActivations[hiddenColumnIndex] * (hiddenSize.z - 1) + 0.5f;
}

void SparseCoder::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* inputCs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    int targetC = (*inputCs)[visibleColumnIndex];

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        float activation = vl.fbWeights.multiplyOHVsT(hiddenCs, visibleIndex, hiddenSize.z) / std::max(1, vl.fbWeights.countT(visibleIndex) / hiddenSize.z);

        vl.inputErrors[visibleIndex] = (vc == targetC ? 1.0f : 0.0f) - sigmoid(activation);
    }
}

void SparseCoder::learnForward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    float error = 0.0f;
    int count = 0;

    {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hiddenCs[hiddenColumnIndex]), hiddenSize);

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            error += vl.fbWeights.multiply(vl.inputErrors, hiddenIndex);
            count += vl.fbWeights.count(hiddenIndex);
        }

        error /= std::max(1, count);
    }

    float delta = alpha * std::tanh(error) * ((hiddenCs[hiddenColumnIndex] + 0.5f) - (hiddenActivations[hiddenColumnIndex] * (hiddenSize.z - 1) + 0.5f));

    delta *= hiddenActivations[hiddenColumnIndex] * (1.0f - hiddenActivations[hiddenColumnIndex]);

    int hiddenIndex = address3(Int3(pos.x, pos.y, 0), Int3(hiddenSize.x, hiddenSize.y, 1));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.ffWeights.deltaOHVs(*inputCs[vli], delta, hiddenIndex, vld.size.z);
    }
}

void SparseCoder::learnBackward(
    const Int2 &pos,
    std::mt19937 &rng,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        float delta = alpha * vl.inputErrors[visibleIndex];

        vl.fbWeights.deltaOHVsT(hiddenCs, delta, visibleIndex, hiddenSize.z);
    }
}

void SparseCoder::initRandom(
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

    std::normal_distribution<float> weightDist(0.0f, 1.0f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, Int3(hiddenSize.x, hiddenSize.y, 1), vld.radius, vl.ffWeights);

        for (int i = 0; i < vl.ffWeights.nonZeroValues.size(); i++)
            vl.ffWeights.nonZeroValues[i] = weightDist(cs.rng);

        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.fbWeights);

        for (int i = 0; i < vl.fbWeights.nonZeroValues.size(); i++)
            vl.fbWeights.nonZeroValues[i] = weightDist(cs.rng);

        // Generate transpose (needed for reconstruction)
        vl.fbWeights.initT();

        vl.inputErrors = FloatBuffer(numVisible, 0.0f);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHiddenColumns, 0.0f);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);

    if (learnEnabled) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            runKernel2(cs, std::bind(SparseCoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs[vli], vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);
        }

        runKernel2(cs, std::bind(SparseCoder::learnForwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            runKernel2(cs, std::bind(SparseCoder::learnBackwardKernel, std::placeholders::_1, std::placeholders::_2, this, vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);
        }
    }
}

void SparseCoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));

    writeBufferToStream(os, &hiddenCs);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.ffWeights);
        writeSMToStream(os, vl.fbWeights);
    }
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));

    readBufferFromStream(is, &hiddenCs);

    hiddenActivations = FloatBuffer(numHiddenColumns, 0.0f);

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

        readSMFromStream(is, vl.ffWeights);
        readSMFromStream(is, vl.fbWeights);

        vl.inputErrors = FloatBuffer(numVisible, 0.0f);
    }
}