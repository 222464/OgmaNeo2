// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"
#include "Predictor.h"
#include "Actor.h"

#include <memory>

namespace ogmaneo {
// Type of hierarchy input layer
enum InputType {
    none = 0,
    prediction = 1
};

// State of hierarchy
struct State {
    std::vector<IntBuffer> hiddenCs;
    std::vector<std::vector<IntBuffer>> predHiddenCs;
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius; // Feed forward radius
        int rRadius; // Recurrent radius, set to -1 to disable
        int pRadius; // Prediction radius

        int historyCapacity; // History buffer capacity

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        ffRadius(2),
        rRadius(2),
        pRadius(2),
        historyCapacity(32)
        {}
    };
private:
    // Layers
    std::vector<SparseCoder> scLayers;
    std::vector<std::vector<std::unique_ptr<Predictor>>> pLayers;
    std::vector<IntBuffer> hiddenCsPrev;

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
        const std::vector<InputType> &inputTypes, // Types of input layers (same size as inputSizes)
        const std::vector<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input layer column states
        const IntBuffer* goalCs, // Goal state
        bool learnEnabled = true // Whether learning is enabled
    );

    // State get
    void getState(
        State &state
    ) const;

    // State set
    void setState(
        const State &state
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
        return scLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        return pLayers.front()[i]->getHiddenCs();
    }

    // Get input sizes
    const std::vector<Int3> &getInputSizes() const {
        return inputSizes;
    }

    // Retrieve a sparse coding layer
    SparseCoder &getSCLayer(
        int l // Layer index
    ) {
        return scLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const SparseCoder &getSCLayer(
        int l // Layer index
    ) const {
        return scLayers[l];
    }

    // Retrieve predictor layer(s)
    std::vector<std::unique_ptr<Predictor>> &getPLayers(
        int l // Layer index
    ) {
        return pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const std::vector<std::unique_ptr<Predictor>> &getPLayers(
        int l // Layer index
    ) const {
        return pLayers[l];
    }
};
} // namespace ogmaneo
