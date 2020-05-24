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

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z) / std::max(1, vl.weights.count(hiddenIndex) / vld.size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    // Learn transitions
    int hiddenIndexMax = address3(Int3(pos.x, pos.y, hiddenCsPrev[hiddenColumnIndex]), hiddenSize);

    transitions.hebbOHVsT(hiddenCs, hiddenIndexMax, hiddenSize.z, beta);
}

void SparseCoder::learnTransition(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // Learn transitions
    int hiddenIndexMax = address3(Int3(pos.x, pos.y, hiddenCsPrev[hiddenColumnIndex]), hiddenSize);

    transitions.hebbOHVsT(hiddenCs, hiddenIndexMax, hiddenSize.z, beta);
}

void SparseCoder::learnFeedForward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* inputCs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    int targetC = (*inputCs)[visibleColumnIndex];

    int maxIndex = 0;
    float maxActivation = -999999.0f;
    std::vector<float> activations(vld.size.z);

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        float sum = vl.weights.multiplyOHVsT(hiddenCs, visibleIndex, hiddenSize.z) / std::max(1, vl.weights.countT(visibleIndex) / hiddenSize.z);

        activations[vc] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = vc;
        }
    }

    if (maxIndex != targetC) {
        for (int vc = 0; vc < vld.size.z; vc++) {
            int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

            float delta = alpha * ((vc == targetC ? 1.0f : 0.0f) - std::exp(activations[vc]));

            vl.weights.deltaChangedOHVsT(hiddenCs, hiddenCsPrev, delta, visibleIndex, hiddenSize.z);
        }
    }
}

void SparseCoder::randomState(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // Random state
    std::uniform_int_distribution<int> columnDist(0, hiddenSize.z - 1);

    hiddenCsRandom[hiddenColumnIndex] = columnDist(rng);
}

void SparseCoder::valueIter(
    const Int2 &pos,
    std::mt19937 &rng,
    const FloatBuffer* rewards
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int randomC = hiddenCsRandom[hiddenColumnIndex];

    int maxIndex = 0;
    float maxValue = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        // Transition probability
        float prob = transitions.multiplyOHVs(hiddenCsRandom, hiddenIndex, hiddenSize.z) / hiddenSize.z;

        float value = prob * ((*rewards)[hiddenIndex] + gamma * hiddenValues[hiddenIndex]);

        if (value > maxValue) {
            maxValue = value;
            maxIndex = hc;
        }
    }
    
    hiddenCsSelect[hiddenColumnIndex] = maxIndex;
    hiddenValues[address3(Int3(pos.x, pos.y, randomC), hiddenSize)] = maxValue;
}

void SparseCoder::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    int transitionRadius,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(-1.0f, 0.0f);

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
    }

    initSMLocalRF(hiddenSize, hiddenSize, transitionRadius, transitions);

    for (int i = 0; i < transitions.nonZeroValues.size(); i++)
        transitions.nonZeroValues[i] = 0.0f;

    transitions.initT();

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);
    hiddenCsPrev = IntBuffer(numHiddenColumns, 0);

    hiddenCsRandom = IntBuffer(numHiddenColumns, 0);
    hiddenCsSelect = IntBuffer(numHiddenColumns, 0);

    hiddenValues = FloatBuffer(numHidden, 0.0f);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

    if (learnEnabled) {
        runKernel2(cs, std::bind(SparseCoder::learnTransitionKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            runKernel2(cs, std::bind(SparseCoder::learnFeedForwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs[vli], vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2);
        }
    }

    // Update prevs
    runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &hiddenCs, &hiddenCsPrev), numHiddenColumns, cs.rng, cs.batchSize1);
}

void SparseCoder::optimize(
    ComputeSystem &cs,
    const FloatBuffer* rewards
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    for (int it = 0; it < valueIters; it++) {
        runKernel2(cs, std::bind(SparseCoder::randomStateKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

        runKernel2(cs, std::bind(SparseCoder::valueIterKernel, std::placeholders::_1, std::placeholders::_2, this, rewards), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
    }

    // Optimize current step to get final selected action (and a free update)
    runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &hiddenCs, &hiddenCsRandom), numHiddenColumns, cs.rng, cs.batchSize1);
    
    runKernel2(cs, std::bind(SparseCoder::valueIterKernel, std::placeholders::_1, std::placeholders::_2, this, rewards), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
}

void SparseCoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));

    writeBufferToStream(os, &hiddenCs);
    writeBufferToStream(os, &hiddenCsPrev);
    
    writeBufferToStream(os, &hiddenCsSelect);

    writeBufferToStream(os, &hiddenValues);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);
    }

    writeSMToStream(os, transitions);
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));

    readBufferFromStream(is, &hiddenCs);
    readBufferFromStream(is, &hiddenCsPrev);

    hiddenCsRandom = IntBuffer(numHiddenColumns, 0);

    readBufferFromStream(is, &hiddenCsSelect);

    readBufferFromStream(is, &hiddenValues);

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
    }

    readSMFromStream(is, transitions);
}