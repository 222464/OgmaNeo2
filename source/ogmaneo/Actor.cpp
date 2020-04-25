// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEOLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

void Actor::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    float maxQ = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        std::vector<float> probs(supportSize);
        float maxProb = -999999.0f;

        for (int zi = 0; zi < supportSize; zi++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc * supportSize + zi), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z * supportSize));

            float sum = 0.0f;
            int count = 0;

            // For each visible layer
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                sum += vl.weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
                count += vl.weights.count(hiddenIndex) / vld.size.z;
            }

            sum /= std::max(1, count);

            probs[zi] = sum;

            maxProb = std::max(maxProb, sum);
        }

        float total = 0.0f;

        for (int zi = 0; zi < supportSize; zi++) {
            probs[zi] = std::exp(probs[zi] - maxProb);

            total += probs[zi];
        }

        float scale = 1.0f / std::max(0.0001f, total);

        for (int zi = 0; zi < supportSize; zi++)
            probs[zi] *= scale;
        
        float q = 0.0f;

        for (int zi = 0; zi < supportSize; zi++) {
            float supportValue = static_cast<float>(zi) / static_cast<float>(supportSize - 1) * 2.0f - 1.0f;

            q += supportValue * probs[zi];
        }

        if (q > maxQ) {
            maxQ = q;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Actor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    int t
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    float rewardSum = 0.0f;
    float g = 1.0f;

    for (int n = 0; n < qSteps; n++) {
        rewardSum += historySamples[t + n]->reward * g;

        g *= gamma;
    }
    
    rewardSum /= qSteps;
    
    const HistorySample &sAhead = *historySamples[t + qSteps - 1];
    const HistorySample &s = *historySamples[t];
    HistorySample &sPrev = *historySamples[t - 1];

    float maxQ = -999999.0f;
    std::vector<float> maxProbs;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        std::vector<float> probs(supportSize);
        float maxProb = -999999.0f;

        for (int zi = 0; zi < supportSize; zi++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc * supportSize + zi), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z * supportSize));

            float sum = 0.0f;
            int count = 0;

            // For each visible layer
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                sum += vl.weights.multiplyOHVs(sAhead.inputCs[vli], hiddenIndex, vld.size.z);
                count += vl.weights.count(hiddenIndex) / vld.size.z;
            }

            sum /= std::max(1, count);

            probs[zi] = sum;

            maxProb = std::max(maxProb, sum);
        }

        float total = 0.0f;

        for (int zi = 0; zi < supportSize; zi++) {
            probs[zi] = std::exp(probs[zi] - maxProb);

            total += probs[zi];
        }

        float scale = 1.0f / std::max(0.0001f, total);

        for (int zi = 0; zi < supportSize; zi++)
            probs[zi] *= scale;
        
        float q = 0.0f;

        for (int zi = 0; zi < supportSize; zi++) {
            float supportValue = static_cast<float>(zi) / static_cast<float>(supportSize - 1) * 2.0f - 1.0f;

            q += supportValue * probs[zi];
        }

        if (q > maxQ) {
            maxQ = q;
            maxProbs = probs;
        }
    }

    std::vector<float> targetProbs(supportSize, 0.0f);
        
    float supportDelta = 2.0f / static_cast<float>(supportSize - 1);

    for (int zi = 0; zi < supportSize; zi++) {
        float supportValue = static_cast<float>(zi) / static_cast<float>(supportSize - 1) * 2.0f - 1.0f;

        float tz = std::min(1.0f, std::max(-1.0f, (1.0f - g) * rewardSum + g * supportValue));

        float b = (tz + 1.0f) / supportDelta;

        int lower = std::floor(b);
        int upper = std::ceil(b);

        // Distribute
        targetProbs[lower] += maxProbs[zi] * (upper - b);
        targetProbs[upper] += maxProbs[zi] * (b - lower);
    }

    // Reduce error
    std::vector<float> probs(supportSize);
    float maxProb = -999999.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, s.hiddenCsPrev[hiddenColumnIndex] * supportSize + zi), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z * supportSize));

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(sPrev.inputCs[vli], hiddenIndex, vld.size.z);
            count += vl.weights.count(hiddenIndex) / vld.size.z;
        }

        sum /= std::max(1, count);

        probs[zi] = sum;

        maxProb = std::max(maxProb, sum);
    }

    float total = 0.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        probs[zi] = std::exp(probs[zi] - maxProb);

        total += probs[zi];
    }

    float scale = 1.0f / std::max(0.0001f, total);

    for (int zi = 0; zi < supportSize; zi++)
        probs[zi] *= scale;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, s.hiddenCsPrev[hiddenColumnIndex] * supportSize + zi), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z * supportSize));

        float delta = alpha * (targetProbs[zi] - probs[zi]);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.weights.deltaOHVs(sPrev.inputCs[vli], delta, hiddenIndex, vld.size.z);
        }
    }
}

void Actor::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    int supportSize,
    int historyCapacity,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    this->supportSize = supportSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z * supportSize), vld.radius, vl.weights);

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i] = std::make_shared<HistorySample>();

        historySamples[i]->inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            historySamples[i]->inputCs[vli] = IntBuffer(numVisibleColumns);
        }

        historySamples[i]->hiddenCsPrev = IntBuffer(numHiddenColumns);
    }
}

const Actor &Actor::operator=(
    const Actor &other
) {
    hiddenSize = other.hiddenSize;

    historySize = other.historySize;

    hiddenCs = other.hiddenCs;

    visibleLayerDescs = other.visibleLayerDescs;
    visibleLayers = other.visibleLayers;

    alpha = other.alpha;
    gamma = other.gamma;
    qSteps = other.qSteps;
    historyIters = other.historyIters;

    historySamples.resize(other.historySamples.size());

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        (*historySamples[t]) = (*other.historySamples[t]);
    }

    return *this;
}

void Actor::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenCsPrev,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    if (historySize == historySamples.size()) {
        // Circular buffer swap
        std::shared_ptr<HistorySample> temp = historySamples.front();

        for (int i = 0; i < historySamples.size() - 1; i++)
            historySamples[i] = historySamples[i + 1];

        historySamples.back() = temp;
    }
    else
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = *historySamples[historySize - 1];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            // Copy visible Cs
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &s.inputCs[vli]), numVisibleColumns, cs.rng, cs.batchSize1);
        }

        // Copy hidden Cs
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, hiddenCsPrev, &s.hiddenCsPrev), numHiddenColumns, cs.rng, cs.batchSize1);

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > qSteps) {
        std::uniform_int_distribution<int> sampleDist(1, historySize - qSteps);

        for (int it = 0; it < historyIters; it++) {
            int t = sampleDist(cs.rng);

            // Learn kernel
            runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, t), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
        }
    }

    // Forward kernel
    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
}

void Actor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&qSteps), sizeof(int));
    os.write(reinterpret_cast<const char*>(&historyIters), sizeof(int));

    os.write(reinterpret_cast<const char*>(&supportSize), sizeof(int));
    os.write(reinterpret_cast<const char*>(&historySize), sizeof(int));

    writeBufferToStream(os, &hiddenCs);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);
    }

    int numHistorySamples = historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = *historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writeBufferToStream(os, &s.inputCs[vli]);

        writeBufferToStream(os, &s.hiddenCsPrev);

        os.write(reinterpret_cast<const char*>(&s.reward), sizeof(float));
    }
}

void Actor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&qSteps), sizeof(int));
    is.read(reinterpret_cast<char*>(&historyIters), sizeof(int));

    is.read(reinterpret_cast<char*>(&supportSize), sizeof(int));
    is.read(reinterpret_cast<char*>(&historySize), sizeof(int));

    readBufferFromStream(is, &hiddenCs);

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

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    historySamples.resize(numHistorySamples);

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        HistorySample &s = *historySamples[t];

        s.inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            readBufferFromStream(is, &s.inputCs[vli]);

        readBufferFromStream(is, &s.hiddenCsPrev);
 
        is.read(reinterpret_cast<char*>(&s.reward), sizeof(float));
    }
}