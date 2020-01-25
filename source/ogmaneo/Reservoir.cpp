// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Reservoir.h"

using namespace ogmaneo;

void Reservoir::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputStates
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = hiddenBiases[hiddenIndex];
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            if (vld.noDiagonal) {
                sum += vl.weights.multiplyNoDiagonal(*inputStates[vli], hiddenIndex);
                count += vl.weights.count(hiddenIndex) - 1;
            }
            else {
                sum += vl.weights.multiply(*inputStates[vli], hiddenIndex);
                count += vl.weights.count(hiddenIndex);
            }
        }

        hiddenStates[hiddenIndex] = std::tanh(sum * std::sqrt(1.0f / std::max(1, count)));
    }
}

void Reservoir::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    float biasScale
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;
    
    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vld.dropRatio, vl.weights, cs.rng);

        std::normal_distribution<float> weightDist(0.0f, vld.scale);

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);
    }

    // Hidden states
    hiddenStates = FloatBuffer(numHidden, 0.0f);
    hiddenStatesPrev = FloatBuffer(numHidden, 0.0f);

    // Biases
    hiddenBiases = FloatBuffer(numHidden);

    std::normal_distribution<float> biasDist(0.0f, biasScale);

    for (int i = 0; i < numHidden; i++)
        hiddenBiases[i] = biasDist(cs.rng);
}

void Reservoir::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputStates
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    // Copy to prev
#ifdef KERNEL_NO_THREAD
    for (int x = 0; x < numHidden; x++)
        copyFloat(x, cs.rng, &hiddenStates, &hiddenStatesPrev);
#else
    runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, &hiddenStates, &hiddenStatesPrev), numHidden, cs.rng, cs.batchSize1);
#endif

#ifdef KERNEL_NO_THREAD
    for (int x = 0; x < hiddenSize.x; x++)
        for (int y = 0; y < hiddenSize.y; y++)
            forward(Int2(x, y), cs.rng, inputStates);
#else
    runKernel2(cs, std::bind(Reservoir::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputStates), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
#endif
}

void Reservoir::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    writeBufferToStream(os, &hiddenStates);
    writeBufferToStream(os, &hiddenStatesPrev);

    writeBufferToStream(os, &hiddenBiases);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);
    }
}

void Reservoir::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    readBufferFromStream(is, &hiddenStates);
    readBufferFromStream(is, &hiddenStatesPrev);

    readBufferFromStream(is, &hiddenBiases);

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
}