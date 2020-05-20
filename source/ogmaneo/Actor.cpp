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
    const std::vector<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenTargetCsPrev,
    float reward,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndexTargetPrev = address3(Int3(pos.x, pos.y, (*hiddenTargetCsPrev)[hiddenColumnIndex]), hiddenSize);

    float qPrev = hiddenValues[hiddenIndexTargetPrev];

    int maxIndex = 0;
    float maxQ = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
            count += vl.weights.count(hiddenIndex) / vld.size.z;
        }

        sum /= std::max(1, count);

        hiddenValues[hiddenIndex] = sum;

        if (sum > maxQ) {
            maxQ = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        float delta = alpha * (reward + gamma * maxQ - qPrev);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.traces.fillOHVs(vl.inputCsPrev, hiddenIndexTargetPrev, vld.size.z, 1.0f);
        }

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                vl.weights.deltaTracedOHVs(vl.traces, delta, hiddenIndex, traceDecay);
            }
        }
    }
}

void Actor::initRandom(
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

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.weights);

        vl.traces = vl.weights;

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++) {
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);
            vl.traces.nonZeroValues[i] = 0.0f;
        }

        vl.inputCsPrev = IntBuffer(numVisibleColumns, 0);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenValues = FloatBuffer(numHidden, 0.0f);
}

void Actor::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenTargetCsPrev,
    float reward,
    bool learnEnabled
) {
    // Forward kernel
    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, hiddenTargetCsPrev, reward, learnEnabled), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

    // Copy
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &vl.inputCsPrev), numVisibleColumns, cs.rng, cs.batchSize1);
    }
}

void Actor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&traceDecay), sizeof(float));

    os.write(reinterpret_cast<const char*>(&historySize), sizeof(int));

    writeBufferToStream(os, &hiddenCs);

    writeBufferToStream(os, &hiddenValues);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);
        writeSMToStream(os, vl.traces);

        writeBufferToStream(os, &vl.inputCsPrev);
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
    is.read(reinterpret_cast<char*>(&traceDecay), sizeof(float));

    is.read(reinterpret_cast<char*>(&historySize), sizeof(int));

    readBufferFromStream(is, &hiddenCs);

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
        readSMFromStream(is, vl.traces);

        readBufferFromStream(is, &vl.inputCsPrev);
    }
}