// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
    const FloatBuffer* feedBackStates,
    const FloatBuffer* inputStates
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = (feedBackStates == nullptr ? 0.0f : visibleLayer.feedBackWeights.multiply(*feedBackStates, hiddenIndex)) + visibleLayer.inputWeights.multiply(*inputStates, hiddenIndex);
        int count = visibleLayer.feedBackWeights.count(hiddenIndex);
    
        hiddenStates[hiddenIndex] = std::tanh(sum * std::sqrt(1.0f / std::max(1, count)));
    }
}

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    int index
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = (historySamples[index]->feedBackStates.empty() ? 0.0f : visibleLayer.feedBackWeights.multiply(historySamples[index]->feedBackStates, hiddenIndex)) + visibleLayer.inputWeights.multiply(historySamples[index]->inputStates, hiddenIndex);
        int count = visibleLayer.feedBackWeights.count(hiddenIndex);

        float predState = std::tanh(sum * std::sqrt(1.0f / std::max(1, count)));

        float delta = alpha * (historySamples[index - 1]->hiddenTargetStates[hiddenIndex] - predState);// * (1.0f - predState * predState);

        if (!historySamples[index]->feedBackStates.empty())
            visibleLayer.feedBackWeights.deltas(historySamples[index]->feedBackStates, delta, hiddenIndex);

        visibleLayer.inputWeights.deltas(historySamples[index]->inputStates, delta, hiddenIndex);
    }
}

void Predictor::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const VisibleLayerDesc &visibleLayerDesc
) {
    this->visibleLayerDesc = visibleLayerDesc;

    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;
    int numVisible = numVisibleColumns * visibleLayerDesc.size.z;

    // Create weight matrix for this visible layer and initialize randomly
    initSMLocalRF(visibleLayerDesc.size, hiddenSize, visibleLayerDesc.radius, visibleLayerDesc.dropRatio, visibleLayer.feedBackWeights, cs.rng);

    visibleLayer.inputWeights = visibleLayer.feedBackWeights;

    std::normal_distribution<float> weightDist(0.0f, visibleLayerDesc.scale);

    for (int i = 0; i < visibleLayer.feedBackWeights.nonZeroValues.size(); i++) {
        visibleLayer.feedBackWeights.nonZeroValues[i] = weightDist(cs.rng);
        visibleLayer.inputWeights.nonZeroValues[i] = weightDist(cs.rng);
    }

    // Hidden
    hiddenStates = FloatBuffer(numHidden, 0.0f);
}

const Predictor &Predictor::operator=(const Predictor &other) {
    visibleLayerDesc = other.visibleLayerDesc;
    visibleLayer = other.visibleLayer;

    hiddenStates = other.hiddenStates;

    historySamples.resize(other.historySamples.size());

    for (int t = 0; t < historySamples.size(); t++)
        historySamples[t] = std::make_shared<HistorySample>(other.historySamples[t]);
}

void Predictor::activate(
    ComputeSystem &cs,
    const FloatBuffer* feedBackStates,
    const FloatBuffer* inputStates
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_NO_THREAD
    for (int x = 0; x < hiddenSize.x; x++)
        for (int y = 0; y < hiddenSize.y; y++)
            forward(Int2(x, y), cs.rng, feedBackStates, inputStates);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, feedBackStates, inputStates), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
#endif
}

void Predictor::learn(
    ComputeSystem &cs,
    const FloatBuffer* feedBackStates,
    const FloatBuffer* inputStates,
    const FloatBuffer* hiddenTargetStates
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    // Add history sample
    std::shared_ptr<HistorySample> sample = std::make_shared<HistorySample>();

    sample->feedBackStates = (feedBackStates == nullptr ? FloatBuffer{} : *feedBackStates);
    sample->inputStates = *inputStates;
    sample->hiddenTargetStates = *hiddenTargetStates;

    historySamples.insert(historySamples.begin(), sample);

    if (historySamples.size() > maxHistorySamples)
        historySamples.resize(maxHistorySamples);

    if (historySamples.size() > 1) {
        std::uniform_int_distribution<int> sampleDist(1, historySamples.size() - 1);

        for (int it = 0; it < historyIters; it++) {
            int index = sampleDist(cs.rng);

            // Learn kernel
#ifdef KERNEL_NO_THREAD
            for (int x = 0; x < hiddenSize.x; x++)
                for (int y = 0; y < hiddenSize.y; y++)
                    learn(Int2(x, y), cs.rng, index);
#else
            runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, index), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
#endif
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
    os.write(reinterpret_cast<const char*>(&maxHistorySamples), sizeof(int));
    os.write(reinterpret_cast<const char*>(&historyIters), sizeof(int));

    writeBufferToStream(os, &hiddenStates);

    os.write(reinterpret_cast<const char*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    writeSMToStream(os, visibleLayer.feedBackWeights);
    writeSMToStream(os, visibleLayer.inputWeights);

    int numHistorySamples = historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        writeBufferToStream(os, &historySamples[t]->feedBackStates);
        writeBufferToStream(os, &historySamples[t]->inputStates);
        writeBufferToStream(os, &historySamples[t]->hiddenTargetStates);
    }
}

void Predictor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&maxHistorySamples), sizeof(int));
    is.read(reinterpret_cast<char*>(&historyIters), sizeof(int));

    readBufferFromStream(is, &hiddenStates);

    is.read(reinterpret_cast<char*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    readSMFromStream(is, visibleLayer.feedBackWeights);
    readSMFromStream(is, visibleLayer.inputWeights);

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    historySamples.resize(numHistorySamples);

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();
        
        readBufferFromStream(is, &historySamples[t]->feedBackStates);
        readBufferFromStream(is, &historySamples[t]->inputStates);
        readBufferFromStream(is, &historySamples[t]->hiddenTargetStates);
    }
}