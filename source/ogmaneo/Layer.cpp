// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Layer.h"

using namespace ogmaneo;

void Layer::forwardMapping(
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

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.mappingWeights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z) / std::max(1, vl.mappingWeights.count(hiddenIndex) / vld.size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Layer::learnMapping(
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

        float sum = vl.mappingWeights.multiplyOHVsT(hiddenCs, visibleIndex, hiddenSize.z) / std::max(1, vl.mappingWeights.countT(visibleIndex) / hiddenSize.z);

        activations[vc] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = vc;
        }
    }

    if (maxIndex != targetC) {
        for (int vc = 0; vc < vld.size.z; vc++) {
            int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

            float delta = alpha * ((vc == targetC ? 1.0f : -1.0f) - std::tanh(activations[vc]));

            vl.mappingWeights.deltaOHVsT(hiddenCs, delta, visibleIndex, hiddenSize.z);
        }
    }
}

void Layer::backwardMapping(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        float sum = vl.mappingWeights.multiplyOHVsT(*hiddenCs, visibleIndex, hiddenSize.z) / std::max(1, vl.mappingWeights.countT(visibleIndex) / hiddenSize.z);

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = vc;
        }
    }

    vl.reconCs[visibleColumnIndex] = maxIndex;
}

void Layer::learnTransition(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* feedBackCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndexTransition = address3(Int3(pos.x, pos.y, hiddenCsPrev[hiddenColumnIndex]), hiddenSize);

    transitionWeights.hebbOHVsT(hiddenCs, hiddenIndexTransition, hiddenSize.z, beta);

    if (!feedBackWeights.nonZeroValues.empty()) {
        assert(feedBackCs != nullptr);

        int hiddenIndexFeedBack = address3(Int3(pos.x, pos.y, (*feedBackCs)[hiddenColumnIndex]), hiddenSize);

        feedBackWeights.hebbOHVsT(hiddenCs, hiddenIndexFeedBack, hiddenSize.z, beta);
    }
}

void Layer::setReward(
    const Int2 &pos,
    std::mt19937 &rng,
    float reward
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    hiddenRewards[address3(Int3(pos.x, pos.y, hiddenCs[hiddenColumnIndex]), hiddenSize)] = reward;
}

void Layer::valueIteration(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // Sample randomly
    std::uniform_int_distribution<int> columnDist(0, hiddenSize.z - 1);

    int hiddenIndex = address3(Int3(pos.x, pos.y, columnDist(rng)), hiddenSize);

    hiddenValues[hiddenIndex] = hiddenRewards[hiddenIndex] + gamma * transitionWeights.multiplyOHVsT(hiddenCs, hiddenValues, hiddenIndex, hiddenSize.z) / std::max(1, transitionWeights.countT(hiddenIndex) / hiddenSize.z);
}

void Layer::determinePolicy(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* feedBackCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum;
        
        if (feedBackWeights.nonZeroValues.empty())
            sum = transitionWeights.multiplyOHVs(hiddenCs, hiddenValues, hiddenIndex, hiddenSize.z) / std::max(1, transitionWeights.countT(hiddenIndex) / hiddenSize.z);
        else {
            assert(feedBackCs != nullptr);

            sum = (transitionWeights.multiplyOHVs(hiddenCs, hiddenValues, hiddenIndex, hiddenSize.z) + feedBackWeights.multiplyOHVs(*feedBackCs, hiddenValues, hiddenIndex, hiddenSize.z)) / std::max(1, transitionWeights.countT(hiddenIndex) / hiddenSize.z + feedBackWeights.countT(hiddenIndex) / hiddenSize.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCsNext[hiddenColumnIndex] = maxIndex;
}

void Layer::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    int lateralRadius,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    bool hasFeedBack
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.mappingWeights);

        for (int i = 0; i < vl.mappingWeights.nonZeroValues.size(); i++)
            vl.mappingWeights.nonZeroValues[i] = dist01(cs.rng);

        // Generate transpose (needed for reconstruction)
        vl.mappingWeights.initT();

        vl.reconCs = IntBuffer(numVisibleColumns, 0);
    }

    initSMLocalRF(hiddenSize, hiddenSize, lateralRadius, transitionWeights);

    for (int i = 0; i < transitionWeights.nonZeroValues.size(); i++)
        transitionWeights.nonZeroValues[i] = dist01(cs.rng);

    transitionWeights.initT();

    if (hasFeedBack) {
        feedBackWeights = transitionWeights;

        for (int i = 0; i < feedBackWeights.nonZeroValues.size(); i++)
            feedBackWeights.nonZeroValues[i] = dist01(cs.rng);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);
    hiddenCsPrev = IntBuffer(numHiddenColumns, 0);
    hiddenCsNext = IntBuffer(numHiddenColumns, 0);

    hiddenRewards = FloatBuffer(numHidden, 0.0f);
    hiddenValues = FloatBuffer(numHidden, 0.0f);
}

void Layer::stepForward(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &hiddenCs, &hiddenCsPrev), hiddenCs.size(), cs.rng, cs.batchSize1);

    runKernel2(cs, std::bind(Layer::forwardMappingKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

    if (learnEnabled) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            runKernel2(cs, std::bind(Layer::learnMappingKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs[vli], vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2);
        }
    }
}

void Layer::stepBackward(
    ComputeSystem &cs,
    const IntBuffer* feedBackCs,
    bool learnEnabled,
    float reward
) {
    runKernel2(cs, std::bind(Layer::learnTransitionKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
    
    runKernel2(cs, std::bind(Layer::setRewardKernel, std::placeholders::_1, std::placeholders::_2, this, reward), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
    
    // Value iteration
    for (int it = 0; it < valueIters; it++)
        runKernel2(cs, std::bind(Layer::valueIterationKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

    runKernel2(cs, std::bind(Layer::determinePolicyKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        runKernel2(cs, std::bind(Layer::backwardMappingKernel, std::placeholders::_1, std::placeholders::_2, this, &hiddenCsNext, vli), Int2(vld.size.x, vld.size.y), cs.rng, cs.batchSize2);
    }
}

void Layer::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&valueIters), sizeof(int));

    writeBufferToStream(os, &hiddenCs);
    writeBufferToStream(os, &hiddenCsPrev);
    writeBufferToStream(os, &hiddenCsNext);

    writeBufferToStream(os, &hiddenRewards);
    writeBufferToStream(os, &hiddenValues);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.mappingWeights);

        writeBufferToStream(os, &vl.reconCs);
    }

    writeSMToStream(os, transitionWeights);
}

void Layer::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&valueIters), sizeof(int));

    readBufferFromStream(is, &hiddenCs);
    readBufferFromStream(is, &hiddenCsPrev);
    readBufferFromStream(is, &hiddenCsNext);

    readBufferFromStream(is, &hiddenRewards);
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

        readSMFromStream(is, vl.mappingWeights);

        readBufferFromStream(is, &vl.reconCs);
    }

    readSMFromStream(is, transitionWeights);
}