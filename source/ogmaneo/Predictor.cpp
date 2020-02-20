// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

void Predictor::forward(
    const Int2 &pos,
    std::mt19937 &rng
) {
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

            sum += vl.weights.multiplyOHVs(vl.inputCs, hiddenIndex, vld.size.z);
            count += vl.weights.count(hiddenIndex) / vld.size.z;
        }

        sum /= std::max(1, count);

        hiddenActivations[hiddenIndex] = sum;

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[address2(pos, Int2(hiddenSize.x, hiddenSize.y))] = maxIndex;
}

void Predictor::error(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenTargetCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenErrors[hiddenIndex] = ((hc == targetC ? 1.0f : -1.0f) - std::tanh(hiddenActivationsPrev[hiddenIndex]));
    }
}

void Predictor::prop(
    const Int2 &pos,
    std::mt19937 &rng,
    FloatBuffer* errors,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    int visibleIndex = address3(Int3(pos.x, pos.y, vl.inputCsPrev[visibleColumnIndex]), vld.size);

    (*errors)[visibleColumnIndex] += vl.weights.multiplyT(hiddenErrors, visibleIndex) / std::max(1, vl.weights.countT(visibleIndex) / hiddenSize.z);
}

void Predictor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenTargetCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float delta = alpha * ((hc == targetC ? 1.0f : -1.0f) - std::tanh(hiddenActivationsPrev[hiddenIndex]));

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.weights.deltaOHVs(vl.inputCsPrev, delta, hiddenIndex, vld.size.z);
        }
    }
}

void Predictor::initRandom(
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

    std::uniform_real_distribution<float> weightDist(-0.01f, 0.01f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.weights);

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);

        if (vld.canPropagate)
            vl.weights.initT();

        vl.inputCs = IntBuffer(numVisibleColumns, 0);
        vl.inputCsPrev = IntBuffer(numVisibleColumns, 0);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenActivationsPrev = FloatBuffer(numHidden, 0.0f);

    hiddenErrors = FloatBuffer(numHidden, 0.0f);
}

void Predictor::activate(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, &hiddenActivations, &hiddenActivationsPrev), numHiddenColumns, cs.rng, cs.batchSize1, cs.pool.size() > 1);

    // Forward kernel
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &vl.inputCs, &vl.inputCsPrev), numVisibleColumns, cs.rng, cs.batchSize1, cs.pool.size() > 1);
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &vl.inputCs), numVisibleColumns, cs.rng, cs.batchSize1, cs.pool.size() > 1);
    }
}

void Predictor::propagate(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs,
    FloatBuffer* errors,
    int vli
) {
    assert(visibleLayerDescs[vli].canPropagate);

    runKernel2(cs, std::bind(Predictor::errorKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenTargetCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);

    runKernel2(cs, std::bind(Predictor::propKernel, std::placeholders::_1, std::placeholders::_2, this, errors, vli), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);
}

void Predictor::learn(
    ComputeSystem &cs,
    const IntBuffer* hiddenTargetCs
) {
    // Learn kernel
    runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenTargetCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2, cs.pool.size() > 1);
}

void Predictor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));

    writeBufferToStream(os, &hiddenCs);

    writeBufferToStream(os, &hiddenActivations);
    writeBufferToStream(os, &hiddenActivationsPrev);

    writeBufferToStream(os, &hiddenErrors);

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

void Predictor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));

    readBufferFromStream(is, &hiddenCs);

    readBufferFromStream(is, &hiddenActivations);
    readBufferFromStream(is, &hiddenActivationsPrev);

    readBufferFromStream(is, &hiddenErrors);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl.weights);

        readBufferFromStream(is, &vl.inputCs);
        readBufferFromStream(is, &vl.inputCsPrev);
    }
}