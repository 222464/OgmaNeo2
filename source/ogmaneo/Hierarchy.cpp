// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace ogmaneo;

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    scLayers.resize(layerDescs.size());
    pLayers.resize(layerDescs.size());
    hiddenCsPrev.resize(layerDescs.size());

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size());

            for (int i = 0; i < inputSizes.size(); i++) {
                scVisibleLayerDescs[i].size = inputSizes[i];
                scVisibleLayerDescs[i].radius = layerDescs[l].ffRadius;
            }

            // Predictors
            pLayers[l].resize(inputSizes.size());

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            pVisibleLayerDesc.radius = layerDescs[l].pRadius;

            // Create predictors
            for (int p = 0; p < pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::prediction) {
                    pLayers[l][p] = std::make_unique<Predictor>();

                    pLayers[l][p]->initRandom(cs, inputSizes[p], layerDescs[l].historyCapacity, pVisibleLayerDesc);
                }
            }
        }
        else {
            scVisibleLayerDescs.resize(1);

            scVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            scVisibleLayerDescs[0].radius = layerDescs[l].ffRadius;

            pLayers[l].resize(1);

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            pVisibleLayerDesc.radius = layerDescs[l].pRadius;

            // Create predictors
            for (int p = 0; p < pLayers[l].size(); p++) {
                pLayers[l][p] = std::make_unique<Predictor>();

                pLayers[l][p]->initRandom(cs, layerDescs[l - 1].hiddenSize, layerDescs[l].historyCapacity, pVisibleLayerDesc);
            }
        }

        if (layerDescs[l].rRadius >= 0) {
            SparseCoder::VisibleLayerDesc vld;

            vld.size = layerDescs[l].hiddenSize;
            vld.radius = layerDescs[l].rRadius;

            scVisibleLayerDescs.push_back(vld);

            hiddenCsPrev[l] = IntBuffer(layerDescs[l].hiddenSize.x * layerDescs[l].hiddenSize.y, 0);
        }
		
        // Create the sparse coding layer
        scLayers[l].initRandom(cs, layerDescs[l].hiddenSize, scVisibleLayerDescs);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    scLayers = other.scLayers;
    hiddenCsPrev = other.hiddenCsPrev;

    inputSizes = other.inputSizes;

    pLayers.resize(other.pLayers.size());

    for (int l = 0; l < scLayers.size(); l++) {
        pLayers[l].resize(other.pLayers[l].size());

        for (int v = 0; v < pLayers[l].size(); v++) {
            if (other.pLayers[l][v] != nullptr) {
                pLayers[l][v] = std::make_unique<Predictor>();

                (*pLayers[l][v]) = (*other.pLayers[l][v]);
            }
            else
                pLayers[l][v] = nullptr;
        }
    }

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    const IntBuffer* goalCs,
    bool learnEnabled
) {
    assert(inputCs.size() == inputSizes.size());

    // Forward
    for (int l = 0; l < scLayers.size(); l++) {
        std::vector<const IntBuffer*> fullLayerInputCs = (l == 0 ? inputCs : std::vector<const IntBuffer*>{ &scLayers[l - 1].getHiddenCs() });

        if (fullLayerInputCs.size() < scLayers[l].getNumVisibleLayers())
            fullLayerInputCs.push_back(&hiddenCsPrev[l]);

        // Activate sparse coder
        scLayers[l].step(cs, fullLayerInputCs, learnEnabled);

        // Copy
        if (!hiddenCsPrev[l].empty())
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &scLayers[l].getHiddenCs(), &hiddenCsPrev[l]), scLayers[l].getHiddenCs().size(), cs.rng, cs.batchSize1);
    }

    // Backward
    for (int l = scLayers.size() - 1; l >= 0; l--) {
        // Feed back is current layer state and next higher layer prediction
        const IntBuffer* feedBackCs;

        if (l < scLayers.size() - 1)
            feedBackCs = &pLayers[l + 1][0]->getHiddenCs();
        else
            feedBackCs = goalCs;

        // Step actor layers
        for (int p = 0; p < pLayers[l].size(); p++) {
            if (pLayers[l][p] != nullptr) {
                if (learnEnabled)
                    pLayers[l][p]->learn(cs, l == 0 ? inputCs[p] : &scLayers[l - 1].getHiddenCs(), &scLayers[l].getHiddenCs());

                pLayers[l][p]->activate(cs, feedBackCs, &scLayers[l].getHiddenCs());
            }
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = scLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(inputSizes.data()), numInputs * sizeof(Int3));

    for (int l = 0; l < numLayers; l++) {
        scLayers[l].writeToStream(os);

        // Predictors
        for (int v = 0; v < pLayers[l].size(); v++) {
            char exists = pLayers[l][v] != nullptr;

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists)
                pLayers[l][v]->writeToStream(os);
        }
    }
}

void Hierarchy::readFromStream(
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;

    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));

    inputSizes.resize(numInputs);

    is.read(reinterpret_cast<char*>(inputSizes.data()), numInputs * sizeof(Int3));

    scLayers.resize(numLayers);
    pLayers.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;

        scLayers[l].readFromStream(is);
        
        pLayers[l].resize(l == 0 ? inputSizes.size() : 1);

        // Predictors
        for (int v = 0; v < pLayers[l].size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                pLayers[l][v] = std::make_unique<Predictor>();
                pLayers[l][v]->readFromStream(is);
            }
            else
                pLayers[l][v] = nullptr;
        }
    }
}

void Hierarchy::getState(
    State &state
) const {
    int numLayers = scLayers.size();

    state.hiddenCs.resize(numLayers);
    state.predHiddenCs.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        state.hiddenCs[l] = scLayers[l].getHiddenCs();

        state.predHiddenCs[l].resize(pLayers[l].size());

        for (int j = 0; j < pLayers[l].size(); j++)
            state.predHiddenCs[l][j] = pLayers[l][j]->getHiddenCs();
    }
}

void Hierarchy::setState(
    const State &state
) {
    int numLayers = scLayers.size();

    for (int l = 0; l < numLayers; l++) {
        scLayers[l].hiddenCs = state.hiddenCs[l];

        for (int j = 0; j < pLayers[l].size(); j++)
            pLayers[l][j]->hiddenCs = state.predHiddenCs[l][j];
    }
}