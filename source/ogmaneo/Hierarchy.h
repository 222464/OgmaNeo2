// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Layer.h"

#include <memory>

namespace ogmaneo {
// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int mRadius; // Feed forward radius
        int lRadius; // Prediction radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        mRadius(2),
        lRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(4)
        {}
    };

private:
    // Layers
    std::vector<Layer> layers;

    // Histories
    std::vector<std::vector<std::shared_ptr<IntBuffer>>> histories;
    std::vector<std::vector<int>> historySizes;

    // Per-layer values
    std::vector<char> updates;

    std::vector<int> ticks;
    std::vector<int> ticksPerUpdate;

    std::vector<float> rewardAccums;
    std::vector<float> rewardCounts;

    // Input dimensions
    std::vector<Int3> inputSizes;

public:
    // Default
    Hierarchy() {}

    // Copy
    Hierarchy(
        const Hierarchy &other // Hierarchy to copy from
    ) {
        *this = other;
    }

    // Assignment
    const Hierarchy &operator=(
        const Hierarchy &other // Hierarchy to assign from
    );
    
    // Create a randomly initialized hierarchy
    void initRandom(
        ComputeSystem &cs, // Compute system
        const std::vector<Int3> &inputSizes, // Sizes of input layers
        const std::vector<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input layer column states
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 1.0f // Optional reward for actor layers
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get the number of layers (layers)
    int getNumLayers() const {
        return layers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        int temporalHorizon = layers.front().getNumVisibleLayers() / inputSizes.size();

        return layers.front().getVisibleLayer(i * temporalHorizon + 0).reconCs;
    }

    // Whether this layer received on update this timestep
    bool getUpdate(
        int l // Layer index
    ) const {
        return updates[l];
    }

    // Get current layer ticks, relative to previous layer
    int getTicks(
        int l // Layer Index
    ) const {
        return ticks[l];
    }

    // Get layer ticks per update, relative to previous layer
    int getTicksPerUpdate(
        int l // Layer Index
    ) const {
        return ticksPerUpdate[l];
    }

    // Get input sizes
    const std::vector<Int3> &getInputSizes() const {
        return inputSizes;
    }

    // Retrieve a sparse coding layer
    Layer &getLayer(
        int l // Layer index
    ) {
        return layers[l];
    }

    // Retrieve a sparse coding layer, const version
    const Layer &getLayer(
        int l // Layer index
    ) const {
        return layers[l];
    }
};
} // namespace ogmaneo
