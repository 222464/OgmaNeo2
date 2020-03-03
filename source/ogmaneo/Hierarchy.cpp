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
            aLayers.resize(inputSizes.size());

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1)
                pVisibleLayerDescs.push_back(pVisibleLayerDescs[0]);

            // Actor visible layer descriptors
            std::vector<Actor::VisibleLayerDesc> aVisibleLayerDescs(1);

            aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            aVisibleLayerDescs[0].radius = layerDescs[l].aRadius;

            if (l < scLayers.size() - 1)
                aVisibleLayerDescs.push_back(aVisibleLayerDescs[0]);

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::prediction) {
                    pLayers[l][p] = std::make_unique<Predictor>();

                    pLayers[l][p]->initRandom(cs, inputSizes[p], pVisibleLayerDescs);
                }
                else if (inputTypes[p] == InputType::action) {
                    aLayers[p] = std::make_unique<Actor>();

                    aLayers[p]->initRandom(cs, inputSizes[p], layerDescs[l].historyCapacity, aVisibleLayerDescs);
                }
            }
        }
        else {
            scVisibleLayerDescs.resize(1);

            scVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            scVisibleLayerDescs[0].radius = layerDescs[l].ffRadius;

            pLayers[l].resize(1);

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1)
                pVisibleLayerDescs.push_back(pVisibleLayerDescs[0]);

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                pLayers[l][p] = std::make_unique<Predictor>();

                pLayers[l][p]->initRandom(cs, layerDescs[l - 1].hiddenSize, pVisibleLayerDescs);
            }
        }
		
        // Create the sparse coding layer
        scLayers[l].initRandom(cs, layerDescs[l].hiddenSize, scVisibleLayerDescs);

        hiddenCsPrev[l] = IntBuffer(layerDescs[l].hiddenSize.x * layerDescs[l].hiddenSize.y, 0);
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

    aLayers.resize(inputSizes.size());
    
    for (int v = 0; v < aLayers.size(); v++) {
        if (other.aLayers[v] != nullptr) {
            aLayers[v] = std::make_unique<Actor>();

            (*aLayers[v]) = (*other.aLayers[v]);
        }
        else
            aLayers[v] = nullptr;
    }

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled,
    float reward
) {
    assert(inputCs.size() == inputSizes.size());

    // Forward
    for (int l = 0; l < scLayers.size(); l++)
        // Activate sparse coder
        scLayers[l].step(cs, l == 0 ? inputCs : std::vector<const IntBuffer*>{ &scLayers[l - 1].getHiddenCs() }, learnEnabled);

    // Backward
    for (int l = scLayers.size() - 1; l >= 0; l--) {
        // Feed back is current layer state and next higher layer prediction
        std::vector<const IntBuffer*> feedBackCs(l < scLayers.size() - 1 ? 2 : 1);

        feedBackCs[0] = &scLayers[l].getHiddenCs();

        if (l < scLayers.size() - 1) {
            assert(pLayers[l + 1][0] != nullptr);

            feedBackCs[1] = &pLayers[l + 1][0]->getHiddenCs();
        }

        // Step actor layers
        for (int p = 0; p < pLayers[l].size(); p++) {
            if (pLayers[l][p] != nullptr) {
                if (learnEnabled)
                    pLayers[l][p]->learn(cs, l == 0 ? inputCs[p] : &scLayers[l].getHiddenCs());

                pLayers[l][p]->activate(cs, feedBackCs);
            }
        }

        if (l == 0) {
            // Step actors
            for (int p = 0; p < aLayers.size(); p++) {
                if (aLayers[p] != nullptr)
                    aLayers[p]->step(cs, feedBackCs, inputCs[p], reward, learnEnabled);
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

        writeBufferToStream(os, &hiddenCsPrev[l]);
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        char exists = aLayers[v] != nullptr;

        os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

        if (exists)
            aLayers[v]->writeToStream(os);
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

    hiddenCsPrev.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
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

        readBufferFromStream(is, &hiddenCsPrev[l]);
    }

    // Actors
    aLayers.resize(inputSizes.size());

    for (int v = 0; v < aLayers.size(); v++) {
        char exists;

        is.read(reinterpret_cast<char*>(&exists), sizeof(char));

        if (exists) {
            aLayers[v] = std::make_unique<Actor>();
            aLayers[v]->readFromStream(is);
        }
        else
            aLayers[v] = nullptr;
    }
}