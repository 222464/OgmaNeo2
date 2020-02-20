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
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    float minActivation = 999999.0f;
    std::vector<float> activations(hiddenSize.z);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(vl.inputCs, hiddenIndex, vld.size.z);
            count += vl.weights.count(hiddenIndex) / vld.size.z;
        }

        sum /= std::max(1, count);

        activations[hc] = sum;

        minActivation = std::min(minActivation, sum);
    }

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        activations[hc] = (activations[hc] - minActivation + 1.0f) * (1.0f - hiddenRefractories[hiddenIndex]);

        if (activations[hc] > maxActivation) {
            maxActivation = activations[hc];

            maxIndex = hc;
        }

        hiddenRefractories[hiddenIndex] *= gamma;
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    hiddenRefractories[address3(Int3(pos.x, pos.y, maxIndex), hiddenSize)] = 1.0f;
}

void SparseCoder::learnForward(
    const Int2 &pos,
    std::mt19937 &rng,
    const FloatBuffer* errors
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, hiddenCsPrev[hiddenColumnIndex]), hiddenSize);

    float delta = alpha * (*errors)[hiddenColumnIndex];

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.weights.deltaOHVs(vl.inputCsPrev, delta, hiddenIndex, vld.size.z);
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
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.weights);

        if (vld.recurrent) {
            for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
                vl.weights.nonZeroValues[i] = 0.0f;
        }
        else {
            for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
                vl.weights.nonZeroValues[i] = weightDist(cs.rng);
        }

        vl.inputCs = IntBuffer(numVisibleColumns, 0);
        vl.inputCsPrev = IntBuffer(numVisibleColumns, 0);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);
    hiddenCsPrev = IntBuffer(numHiddenColumns, 0);

    hiddenRefractories = FloatBuffer(numHidden, 0.0f);
}

void SparseCoder::activate(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &hiddenCs, &hiddenCsPrev), numHiddenColumns, cs.rng, cs.batchSize1, cs.pool.size() > 1);

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &vl.inputCs, &vl.inputCsPrev), numVisibleColumns, cs.rng, cs.batchSize1, cs.pool.size() > 1);
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &vl.inputCs), numVisibleColumns, cs.rng, cs.batchSize1, cs.pool.size() > 1);
    }

    runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);
}

void SparseCoder::learn(
    ComputeSystem &cs,
    const FloatBuffer* errors
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    runKernel2(cs, std::bind(SparseCoder::learnForwardKernel, std::placeholders::_1, std::placeholders::_2, this, errors), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);
}

void SparseCoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));

    writeBufferToStream(os, &hiddenCs);
    writeBufferToStream(os, &hiddenCsPrev);

    writeBufferToStream(os, &hiddenRefractories);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);

        writeBufferToStream(os, &vl.inputCs);
        writeBufferToStream(os, &vl.inputCsPrev);
    }
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));

    readBufferFromStream(is, &hiddenCs);
    readBufferFromStream(is, &hiddenCsPrev);

    readBufferFromStream(is, &hiddenRefractories);

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

        readBufferFromStream(is, &vl.inputCs);
        readBufferFromStream(is, &vl.inputCsPrev);
    }
}