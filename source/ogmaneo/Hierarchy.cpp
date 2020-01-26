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
    eLayers.resize(layerDescs.size());
    pLayers.resize(layerDescs.size());
    pErrors.resize(layerDescs.size());

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        std::vector<Reservoir::VisibleLayerDesc> rVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            rVisibleLayerDescs.resize(inputSizes.size());

            for (int i = 0; i < inputSizes.size(); i++) {
                rVisibleLayerDescs[i].size = inputSizes[i];
                rVisibleLayerDescs[i].radius = layerDescs[l].rfRadius;
                rVisibleLayerDescs[i].scale = layerDescs[l].rfScale;
                rVisibleLayerDescs[i].dropRatio = layerDescs[l].rfDropRatio;
                rVisibleLayerDescs[i].noDiagonal = false;
            }

            // Predictors
            pLayers[l].resize(inputSizes.size());
            pErrors[l].resize(inputSizes.size());

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            pVisibleLayerDesc.radius = layerDescs[l].pRadius;
            pVisibleLayerDesc.scale = layerDescs[l].pScale;
            pVisibleLayerDesc.dropRatio = layerDescs[l].pDropRatio;

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                pLayers[l][p].initRandom(cs, inputSizes[p], pVisibleLayerDesc);

                pErrors[l][p] = FloatBuffer(inputSizes[p].x * inputSizes[p].y * inputSizes[p].z, 0.0f);
            }
        }
        else {
            rVisibleLayerDescs.resize(1);

            rVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            rVisibleLayerDescs[0].radius = layerDescs[l].rfRadius;
            rVisibleLayerDescs[0].scale = layerDescs[l].rfScale;
            rVisibleLayerDescs[0].dropRatio = layerDescs[l].rfDropRatio;
            rVisibleLayerDescs[0].noDiagonal = false;

            pLayers[l].resize(1);
            pErrors[l].resize(1);

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            pVisibleLayerDesc.radius = layerDescs[l].pRadius;
            pVisibleLayerDesc.scale = layerDescs[l].pScale;
            pVisibleLayerDesc.dropRatio = layerDescs[l].pDropRatio;

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                pLayers[l][p].initRandom(cs, layerDescs[l - 1].hiddenSize, pVisibleLayerDesc);

                pErrors[l][p] = FloatBuffer(layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y * layerDescs[l - 1].hiddenSize.z, 0.0f);
            }
        }

        eLayers[l].initRandom(cs, layerDescs[l].hiddenSize, rVisibleLayerDescs, layerDescs[l].rbScale);

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
        std::vector<const FloatBuffer*> layerInputs = (l == 0 ? inputStates : std::vector<const FloatBuffer*>{ &eLayers[l - 1].getHiddenStates() });

        std::vector<const FloatBuffer*> layerErrors(inputStates.size());

        // Find differences
        for (int p = 0; p < pErrors[l].size(); p++) {
            // Difference
#ifdef KERNEL_NO_THREAD
            for (int x = 0; x < pErrors[l][p].size(); x++)
                diffFloat(x, cs.rng, layerInputs[p], &pLayers[l][p].getHiddenStates(), &pErrors[l][p]);
#else
            runKernel1(cs, std::bind(diffFloat, std::placeholders::_1, std::placeholders::_2, layerInputs[p], &pLayers[l][p].getHiddenStates(), &pErrors[l][p]), pErrors[l][p].size(), cs.rng, cs.batchSize1);
#endif

            layerErrors[p] = &pErrors[l][p];
        }

        // Add recurrent if needed
        if (layerInputs.size() < rLayers[l].getNumVisibleLayers())
            layerInputs.push_back(&rLayers[l].getHiddenStatesPrev());

        rLayers[l].step(cs, layerInputs);
        eLayers[l].step(cs, layerErrors);
    }

    // Backward
    for (int l = rLayers.size() - 1; l >= 0; l--) {
        const FloatBuffer* feedBackStates = l < rLayers.size() - 1 ? &pLayers[l + 1][0].getHiddenStates() : nullptr;

        // Step actor layers
        for (int p = 0; p < pLayers[l].size(); p++) {
            if (learnEnabled) 
                pLayers[l][p].learn(cs, feedBackStates, &rLayers[l].getHiddenStates(), l == 0 ? inputStates[p] : &eLayers[l - 1].getHiddenStates());

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
        eLayers[l].writeToStream(os);

        for (int p = 0; p < pLayers[l].size(); p++) {
            pLayers[l][p].writeToStream(os);

            writeBufferToStream(os, &pErrors[l][p]);
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

    rLayers.resize(numLayers);
    eLayers.resize(numLayers);
    pLayers.resize(numLayers);
    pErrors.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        rLayers[l].readFromStream(is);
        eLayers[l].readFromStream(is);

        pLayers[l].resize(l == 0 ? inputSizes.size() : 1);
        pErrors[l].resize(pLayers[l].size());

        for (int p = 0; p < pLayers[l].size(); p++) {
            pLayers[l][p].readFromStream(is);

            readBufferFromStream(is, &pErrors[l][p]);
        }
    }
}