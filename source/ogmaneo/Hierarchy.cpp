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
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    rLayers.resize(layerDescs.size());
    pLayers.resize(layerDescs.size());

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        std::vector<Reservoir::VisibleLayerDesc> rVisibleLayerDescs;
        std::vector<Reservoir::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            rVisibleLayerDescs.resize(inputSizes.size());
            eVisibleLayerDescs.resize(inputSizes.size());

            for (int i = 0; i < inputSizes.size(); i++) {
                rVisibleLayerDescs[i].size = inputSizes[i];
                rVisibleLayerDescs[i].radius = layerDescs[l].rfRadius;
                rVisibleLayerDescs[i].scale = layerDescs[l].rfScale;
                rVisibleLayerDescs[i].dropRatio = layerDescs[l].rfDropRatio;
                rVisibleLayerDescs[i].noDiagonal = false;

                eVisibleLayerDescs[i].size = inputSizes[i];
                eVisibleLayerDescs[i].radius = layerDescs[l].efRadius;
                eVisibleLayerDescs[i].scale = layerDescs[l].efScale;
                eVisibleLayerDescs[i].dropRatio = layerDescs[l].efDropRatio;
                eVisibleLayerDescs[i].noDiagonal = false;
            }

            // Predictors
            pLayers[l].resize(inputSizes.size());

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            pVisibleLayerDesc.radius = layerDescs[l].pRadius;
            pVisibleLayerDesc.scale = layerDescs[l].pScale;
            pVisibleLayerDesc.dropRatio = layerDescs[l].pDropRatio;

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++)
                pLayers[l][p].initRandom(cs, inputSizes[p], pVisibleLayerDesc);
        }
        else {
            rVisibleLayerDescs.resize(1);
            eVisibleLayerDescs.resize(1);

            rVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            rVisibleLayerDescs[0].radius = layerDescs[l].rfRadius;
            rVisibleLayerDescs[0].scale = layerDescs[l].rfScale;
            rVisibleLayerDescs[0].dropRatio = layerDescs[l].rfDropRatio;
            rVisibleLayerDescs[0].noDiagonal = false;

            eVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            eVisibleLayerDescs[0].radius = layerDescs[l].efRadius;
            eVisibleLayerDescs[0].scale = layerDescs[l].efScale;
            eVisibleLayerDescs[0].dropRatio = layerDescs[l].efDropRatio;
            eVisibleLayerDescs[0].noDiagonal = false;

            pLayers[l].resize(1);

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            pVisibleLayerDesc.radius = layerDescs[l].pRadius;
            pVisibleLayerDesc.scale = layerDescs[l].pScale;
            pVisibleLayerDesc.dropRatio = layerDescs[l].pDropRatio;

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++)
                pLayers[l][p].initRandom(cs, layerDescs[l - 1].hiddenSize, pVisibleLayerDesc);
        }

        // Recurrent
        if (layerDescs[l].rrRadius != -1) {
            Reservoir::VisibleLayerDesc vld;

            vld.size = layerDescs[l].hiddenSize;
            vld.radius = layerDescs[l].rrRadius;
            vld.scale = layerDescs[l].rrScale;
            vld.dropRatio = layerDescs[l].rrDropRatio;
            vld.noDiagonal = true;

            rVisibleLayerDescs.push_back(vld);
        }
		
        // Create the sparse coding layer
        rLayers[l].initRandom(cs, layerDescs[l].hiddenSize, rVisibleLayerDescs, layerDescs[l].rbScale);
    }
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputStates,
    bool learnEnabled
) {
    assert(inputStates.size() == inputSizes.size());

    // Forward
    for (int l = 0; l < rLayers.size(); l++) {
        std::vector<const FloatBuffer*> layerInputs = (l == 0 ? inputStates : std::vector<const FloatBuffer*>{ &rLayers[l - 1].getHiddenStates() });

        // Add recurrent if needed
        if (layerInputs.size() < rLayers[l].getNumVisibleLayers())
            layerInputs.push_back(&rLayers[l].getHiddenStatesPrev());

        rLayers[l].step(cs, layerInputs);
    }

    // Backward
    for (int l = rLayers.size() - 1; l >= 0; l--) {
        const FloatBuffer* feedBackStates = l < rLayers.size() - 1 ? &pLayers[l + 1][0].getHiddenStates() : nullptr;

        // Step actor layers
        for (int p = 0; p < pLayers[l].size(); p++) {
            if (learnEnabled) 
                pLayers[l][p].learn(cs, feedBackStates, &rLayers[l].getHiddenStates(), l == 0 ? inputStates[p] : &rLayers[l - 1].getHiddenStates());

            pLayers[l][p].activate(cs, feedBackStates, &rLayers[l].getHiddenStates());
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = rLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(inputSizes.data()), numInputs * sizeof(Int3));

    for (int l = 0; l < numLayers; l++) {
        rLayers[l].writeToStream(os);

        for (int p = 0; p < pLayers[l].size(); p++)
            pLayers[l][p].writeToStream(os);
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

    rLayers.resize(numLayers);
    pLayers.resize(numLayers);
    for (int l = 0; l < numLayers; l++) {
        rLayers[l].readFromStream(is);

        pLayers[l].resize(l == 0 ? inputSizes.size() : 1);

        for (int p = 0; p < pLayers[l].size(); p++)
            pLayers[l][p].readFromStream(is);
    }
}