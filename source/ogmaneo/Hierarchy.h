// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Reservoir.h"
#include "Predictor.h"

#include <memory>

namespace ogmaneo {
// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int rfRadius; // Revervior forward radius
        int rrRadius; // Reservoir recurrent radius
        int pRadius; // Predictor radius

        float rfScale;
        float rfDropRatio;
        float rrScale;
        float rrDropRatio;
        float rbScale;

        float pScale;
        float pDropRatio;

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        rfRadius(2),
        rrRadius(2),
        pRadius(2),
        rfScale(1.0f),
        rfDropRatio(0.0f),
        rrScale(1.0f),
        rrDropRatio(0.0f),
        rbScale(0.1f),
        pScale(0.001f),
        pDropRatio(0.0f)
        {}
    };
private:
    // Layers
    std::vector<Reservoir> rLayers;
    std::vector<Reservoir> eLayers;
    std::vector<std::vector<Predictor>> pLayers;
    std::vector<std::vector<FloatBuffer>> pErrors;
    
    // Input dimensions
    std::vector<Int3> inputSizes;

public:
    // Default
    Hierarchy() {}
    
    // Create a randomly initialized hierarchy
    void initRandom(
        ComputeSystem &cs, // Compute system
        const std::vector<Int3> &inputSizes, // Sizes of input layers
        const std::vector<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputStates, // Input layer column states
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
        return rLayers.size();
    }

    // Retrieve predictions
    const FloatBuffer &getPredictionStates(
        int i // Index of input layer to get predictions for
    ) const {
        return pLayers.front()[i].getHiddenStates();
    }

    // Get input sizes
    const std::vector<Int3> &getInputSizes() const {
        return inputSizes;
    }

    // Retrieve a reservior layer
    Reservoir &getRLayer(
        int l // Layer index
    ) {
        return rLayers[l];
    }

    // Retrieve a reservior layer, const version
    const Reservoir &getRLayer(
        int l // Layer index
    ) const {
        return rLayers[l];
    }

    // Retrieve predictor layer(s)
    std::vector<Predictor> &getPLayers(
        int l // Layer index
    ) {
        return pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const std::vector<Predictor> &getPLayers(
        int l // Layer index
    ) const {
        return pLayers[l];
    }
};
} // namespace ogmaneo
