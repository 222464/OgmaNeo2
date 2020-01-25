// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
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
        Int3 _hiddenSize; // Size of hidden layer

        int _rfRadius; // Revervior forward radius
        int _rrRadius; // Reservoir recurrent radius
        int _pRadius; // Predictor radius

        float _rfScale;
        float _rfDropRatio;
        float _rrScale;
        float _rrDropRatio;
        float _rbScale;

        float _pScale;
        float _pDropRatio;

        LayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _rfRadius(2),
        _rrRadius(2),
        _pRadius(2),
        _rfScale(1.0f),
        _rfDropRatio(0.0f),
        _rrScale(1.0f),
        _rrDropRatio(0.5f),
        _rbScale(0.1f),
        _pScale(1.0f),
        _pDropRatio(0.0f)
        {}
    };
private:
    // Layers
    std::vector<Reservoir> _rLayers;
    std::vector<std::vector<Predictor>> _pLayers;
    std::vector<std::vector<FloatBuffer>> _pErrors;
    
    // Input dimensions
    std::vector<Int3> _inputSizes;

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
        return _rLayers.size();
    }

    // Retrieve predictions
    const FloatBuffer &getPredictionStates(
        int i // Index of input layer to get predictions for
    ) const {
        return _pLayers.front()[i].getHiddenStates();
    }

    // Get input sizes
    const std::vector<Int3> &getInputSizes() const {
        return _inputSizes;
    }

    // Retrieve a reservior layer
    Reservoir &getRLayer(
        int l // Layer index
    ) {
        return _rLayers[l];
    }

    // Retrieve a reservior layer, const version
    const Reservoir &getRLayer(
        int l // Layer index
    ) const {
        return _rLayers[l];
    }

    // Retrieve predictor layer(s)
    std::vector<Predictor> &getPLayers(
        int l // Layer index
    ) {
        return _pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const std::vector<Predictor> &getPLayers(
        int l // Layer index
    ) const {
        return _pLayers[l];
    }
};
} // namespace ogmaneo
