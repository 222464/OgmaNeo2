// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"

#include <memory>

namespace ogmaneo {
// Type of hierarchy input layer
enum InputType {
    _none = 0,
    _predict = 1
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 _hiddenSize; // Size of hidden layer

        // Sparse coder radii
        int _ffRadius; // Feed forward
        int _fbRadius; // Feed back
        int _pRadius; // Prediction

        int _ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)

        int _temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to _ticksPerUpdate

        LayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _ffRadius(2),
        _fbRadius(2),
        _pRadius(2),
        _ticksPerUpdate(2),
        _temporalHorizon(2)
        {}
    };
private:
    // Layers
    std::vector<SparseCoder> _scLayers;

    // Histories
    std::vector<std::vector<std::shared_ptr<IntBuffer>>> _histories;
    std::vector<std::vector<int>> _historySizes;

    // Per-layer values
    std::vector<char> _updates;

    std::vector<int> _ticks;
    std::vector<int> _ticksPerUpdate;

    // Input dimensions
    std::vector<Int3> _inputSizes;

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
        const std::vector<InputType> &inputTypes, // Types of input layers (same size as inputSizes)
        const std::vector<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input layer column states
        const IntBuffer* topFeedBackCs, // Topmost feed back layer
        bool learnEnabled = true // Whether learning is enabled
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get the number of layers (scLayers)
    int getNumLayers() const {
        return _scLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        int index = _scLayers.front().getNumVisibleLayers() - 1 - (_inputSizes.size() - 1 - i);

        return _scLayers.front().getVisibleLayer(index)._reconCs;
    }

    // Whether this layer received on update this timestep
    bool getUpdate(
        int l // Layer index
    ) const {
        return _updates[l];
    }

    // Get current layer ticks, relative to previous layer
    int getTicks(
        int l // Layer Index
    ) const {
        return _ticks[l];
    }

    // Get layer ticks per update, relative to previous layer
    int getTicksPerUpdate(
        int l // Layer Index
    ) const {
        return _ticksPerUpdate[l];
    }

    // Get input sizes
    const std::vector<Int3> &getInputSizes() const {
        return _inputSizes;
    }

    // Retrieve a sparse coding layer
    SparseCoder &getSCLayer(
        int l // Layer index
    ) {
        return _scLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const SparseCoder &getSCLayer(
        int l // Layer index
    ) const {
        return _scLayers[l];
    }
};
} // namespace ogmaneo
