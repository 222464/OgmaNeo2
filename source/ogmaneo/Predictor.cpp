// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"
#include <iostream>

using namespace ogmaneo;

void Predictor::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* goalCs,
    const IntBuffer* inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = hiddenCs[hiddenColumnIndex];
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = weights.multiplyOHVs(*goalCs, hiddenIndex, visibleLayerDesc.size.z) -
            weights.multiplyOHVs(*inputCs, hiddenIndex, visibleLayerDesc.size.z);

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenTargetCs,
    const IntBuffer* inputCsGoal,
    const IntBuffer* inputCs,
    const IntBuffer* inputCsPrev,
    float closeness
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = weights.multiplyOHVs(*inputCsGoal, hiddenIndex, visibleLayerDesc.size.z) -
            weights.multiplyOHVs(*inputCsPrev, hiddenIndex, visibleLayerDesc.size.z);

        sum /= std::max(1, weights.count(hiddenIndex) / visibleLayerDesc.size.z);
            
        float delta = alpha * closeness * closeness * ((hc == targetC ? 1.0f : -1.0f) - std::tanh(sum));

        weights.deltaOHVs(*inputCsGoal, delta, hiddenIndex, visibleLayerDesc.size.z);
        weights.deltaOHVs(*inputCsPrev, -delta, hiddenIndex, visibleLayerDesc.size.z);
    }
}

void Predictor::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    int historyCapacity,
    const VisibleLayerDesc &visibleLayerDesc
) {
    this->visibleLayerDesc = visibleLayerDesc;

    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    // Create weight matrix for this visible layer and initialize randomly
    initSMLocalRF(visibleLayerDesc.size, hiddenSize, visibleLayerDesc.radius, weights);

    for (int i = 0; i < weights.nonZeroValues.size(); i++)
        weights.nonZeroValues[i] = weightDist(cs.rng);

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i] = std::make_shared<HistorySample>();

        historySamples[i]->inputCs = IntBuffer(numVisibleColumns);

        historySamples[i]->hiddenTargetCs = IntBuffer(numHiddenColumns);
    }
}

const Predictor &Predictor::operator=(
    const Predictor &other
) {
    visibleLayerDesc = other.visibleLayerDesc;
    
    hiddenSize = other.hiddenSize;

    alpha = other.alpha;
    historyIters = other.historyIters;
    maxDistance = other.maxDistance;

    weights = other.weights;

    hiddenCs = other.hiddenCs;

    historySize = other.historySize;

    historySamples.resize(other.historySamples.size());

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i] = std::make_shared<HistorySample>();

        *historySamples[i] = *other.historySamples[i];
    }

    return *this;
}

void Predictor::activate(
    ComputeSystem &cs,
    const IntBuffer* goalCs,
    const IntBuffer* inputCs
) {
    // Forward kernel
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, goalCs, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
}

void Predictor::learn(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs,
    const IntBuffer* inputCs
) {
    // Add sample
    if (historySize == historySamples.size()) {
        // Circular buffer swap
        std::shared_ptr<HistorySample> temp = historySamples.front();

        for (int i = 0; i < historySamples.size() - 1; i++)
            historySamples[i] = historySamples[i + 1];

        historySamples.back() = temp;
    }

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = *historySamples[historySize - 1];

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, hiddenTargetCs, &s.hiddenTargetCs), hiddenTargetCs->size(), cs.rng, cs.batchSize1);

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs, &s.inputCs), inputCs->size(), cs.rng, cs.batchSize1);
    }

    if (historySize > 1) {
        std::uniform_int_distribution<int> historyDist(0, historySize - 2);

        for (int it = 0; it < historyIters; it++) {
            int t = historyDist(cs.rng);

            std::uniform_int_distribution<int> distDist(1, std::min(historySize - 1 - t, maxDistance));

            int dist = distDist(cs.rng);

            const HistorySample &sDist = *historySamples[t + dist];
            const HistorySample &s = *historySamples[t + 1];
            const HistorySample &sPrev = *historySamples[t];

            float closeness = static_cast<float>(maxDistance - dist) / static_cast<float>(maxDistance - 1);

            // Learn kernel
            runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s.hiddenTargetCs, &sDist.inputCs, &s.inputCs, &sPrev.inputCs, closeness), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
        }
    }
}

void Predictor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&historyIters), sizeof(int));
    os.write(reinterpret_cast<const char*>(&maxDistance), sizeof(int));

    writeBufferToStream(os, &hiddenCs);

    os.write(reinterpret_cast<const char*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    writeSMToStream(os, weights);

    os.write(reinterpret_cast<const char*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = *historySamples[t];

        writeBufferToStream(os, &s.hiddenTargetCs);
        writeBufferToStream(os, &s.inputCs);
    }
}

void Predictor::readFromStream(
    std::istream &is
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&historyIters), sizeof(int));
    is.read(reinterpret_cast<char*>(&maxDistance), sizeof(int));

    readBufferFromStream(is, &hiddenCs);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    is.read(reinterpret_cast<char*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    readSMFromStream(is, weights);

    is.read(reinterpret_cast<char*>(&historySize), sizeof(int));

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    historySamples.resize(numHistorySamples);

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        HistorySample &s = *historySamples[t];

        readBufferFromStream(is, &s.hiddenTargetCs);
        readBufferFromStream(is, &s.inputCs);
    }
}