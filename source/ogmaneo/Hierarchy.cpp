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
    layers.resize(layerDescs.size());

    ticks.resize(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    historySizes.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

    rewardAccums.resize(layerDescs.size(), 0.0f);
    rewardCounts.resize(layerDescs.size(), 0.0f);

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l].ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Histories for all input layers or just the one sparse coder (if not the first layer)
        histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l].temporalHorizon : layerDescs[l].temporalHorizon);

        historySizes[l].resize(histories[l].size());
		
        // Create sparse coder visible layer descriptors
        std::vector<Layer::VisibleLayerDesc> lVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            lVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    lVisibleLayerDescs[index].size = inputSizes[i];
                    lVisibleLayerDescs[index].radius = layerDescs[l].mRadius;
                }
            }
            
            // Initialize history buffers
			for (int v = 0; v < histories[l].size(); v++) {
				int i = v / layerDescs[l].temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                historySizes[l][v] = inSize;
			}
        }
        else {
            lVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                lVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                lVisibleLayerDescs[t].radius = layerDescs[l].mRadius;
            }

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

			for (int v = 0; v < histories[l].size(); v++) {
                histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                historySizes[l][v] = inSize;
            }
        }
		
        // Create the sparse coding layer
        layers[l].initRandom(cs, layerDescs[l].hiddenSize, layerDescs[l].lRadius, lVisibleLayerDescs, l < layerDescs.size() - 1);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    layers = other.layers;

    inputSizes = other.inputSizes;
    historySizes = other.historySizes;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    updates = other.updates;
    rewardAccums = other.rewardAccums;
    rewardCounts = other.rewardCounts;

    histories.resize(other.histories.size());

    for (int l = 0; l < layers.size(); l++) {
        histories[l].resize(other.histories[l].size());

        for (int v = 0; v < histories[l].size(); v++) {
            histories[l][v] = std::make_unique<IntBuffer>();
            
            (*histories[l][v]) = (*other.histories[l][v]);
        }
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

    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    {
        int temporalHorizon = histories.front().size() / inputSizes.size();

        std::vector<std::shared_ptr<IntBuffer>> lasts(inputSizes.size());
        
        for (int i = 0; i < inputSizes.size(); i++)
            lasts[i] = histories.front()[temporalHorizon - 1 + temporalHorizon * i];
  
        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < inputSizes.size(); i++) {
                // Shift
                histories.front()[t + temporalHorizon * i] = histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < inputSizes.size(); i++) {
            assert(inputSizes[i].x * inputSizes[i].y == inputCs[i]->size());
            
            // Copy
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[i], lasts[i].get()), inputCs[i]->size(), cs.rng, cs.batchSize1);

            histories.front()[0 + temporalHorizon * i] = lasts[i];
        }
    }

    // Set all updates to no update, will be set to true if an update occurred later
    updates.clear();
    updates.resize(layers.size(), false);

    // Forward
    for (int l = 0; l < layers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            // Activate sparse coder
            layers[l].stepForward(cs, constGet(histories[l]), learnEnabled);

            // Add to next layer's history
            if (l < layers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = histories[lNext].size();

                std::shared_ptr<IntBuffer> last = histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    histories[lNext][t] = histories[lNext][t - 1];

                // Copy
                runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &layers[l].getHiddenCs(), last.get()), layers[l].getHiddenCs().size(), cs.rng, cs.batchSize1);

                histories[lNext].front() = last;

                ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = layers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            const IntBuffer* feedBackCs;

            if (l < layers.size() - 1) {
                feedBackCs = &layers[l + 1].getVisibleLayer(ticksPerUpdate[l + 1] - 1 - ticks[l + 1]).reconCs;
            }
            else
                feedBackCs = nullptr;

            layers[l].stepBackward(cs, feedBackCs, learnEnabled, reward);
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = layers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(updates.data()), updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(ticks.data()), ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(ticksPerUpdate.data()), ticksPerUpdate.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(rewardAccums.data()), rewardAccums.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(rewardCounts.data()), rewardCounts.size() * sizeof(float));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < historySizes[l].size(); i++)
            writeBufferToStream(os, histories[l][i].get());

        layers[l].writeToStream(os);
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

    layers.resize(numLayers);

    ticks.resize(numLayers);

    histories.resize(numLayers);
    historySizes.resize(numLayers);
    
    ticksPerUpdate.resize(numLayers);

    updates.resize(numLayers);

    rewardAccums.resize(numLayers);
    rewardCounts.resize(numLayers);

    is.read(reinterpret_cast<char*>(updates.data()), updates.size() * sizeof(char));
    is.read(reinterpret_cast<char*>(ticks.data()), ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(ticksPerUpdate.data()), ticksPerUpdate.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(rewardAccums.data()), rewardAccums.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(rewardCounts.data()), rewardCounts.size() * sizeof(float));
    
    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;
        
        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));
        historySizes[l].resize(numHistorySizes);
        is.read(reinterpret_cast<char*>(historySizes[l].data()), numHistorySizes * sizeof(int));

        histories[l].resize(numHistorySizes);

        for (int i = 0; i < historySizes[l].size(); i++) {
            histories[l][i] = std::make_shared<IntBuffer>();

            readBufferFromStream(is, histories[l][i].get());
        }

        layers[l].readFromStream(is);
    }
}