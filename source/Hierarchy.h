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
    _act = 1
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 _hiddenSize; // Size of hidden layer

        int _scRadius; // Sparse coder radius
        int _rRadius; // Prediction Radius

        int _ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)

        int _temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to _ticksPerUpdate

        LayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _scRadius(2),
        _rRadius(2),
        _ticksPerUpdate(2),
        _temporalHorizon(2)
        {}
    };

    struct RouteLayer {
        std::vector<SparseMatrix> _weights;
        
        std::vector<IntBuffer> _visibleCounts;

        std::vector<FloatBuffer> _errors;
        std::vector<FloatBuffer> _activations;

        IntBuffer _hiddenCounts;
    };

    struct HistorySample {
        std::vector<IntBuffer> _states;

        std::vector<IntBuffer> _actionsPrev;

        float _reward;
    };

private:
    // Layers
    std::vector<SparseCoder> _scLayers;
    std::vector<RouteLayer> _rLayers;

    std::vector<FloatBuffer> _qs;
    std::vector<IntBuffer> _actions;

    // Histories
    std::vector<std::vector<std::shared_ptr<IntBuffer>>> _histories;
    std::vector<std::vector<int>> _historySizes;

    // Per-layer values
    std::vector<char> _updates;

    std::vector<int> _ticks;
    std::vector<int> _ticksPerUpdate;

    // Input dimensions
    std::vector<Int3> _inputSizes;

    // History samples
    std::vector<HistorySample> _historySamples;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int l,
        int vli,
        const IntBuffer* inputCs
    );

    void backward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int l,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int l,
        int vli,
        const IntBuffer* inputCs
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Hierarchy* h,
        const IntBuffer* hiddenCs,
        int l,
        int vli,
        const IntBuffer* inputCs
    ) {
        h->forward(pos, rng, hiddenCs, l, vli, inputCs);
    }

    static void backwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Hierarchy* h,
        const IntBuffer* hiddenCs,
        int l,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        h->backward(pos, rng, hiddenCs, l, inputCs);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Hierarchy* h,
        const IntBuffer* hiddenCs,
        int l,
        int vli,
        const IntBuffer* inputCs
    ) {
        h->learn(pos, rng, hiddenCs, l, vli, inputCs);
    }

public:
    float _alpha; // Routing learning rate
    float _gamma; // Discount factor
    float _clip; // Gradient clip

    int _maxHistorySamples; // Maximum number of history samples
    int _historyIters; // Number of times to iterate over history

    // Default
    Hierarchy()
    :
    _alpha(0.1f),
    _gamma(0.99f),
    _clip(0.1f),
    _maxHistorySamples(32),
    _historyIters(4)
    {}

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
        float reward, // Reinforcement signal
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
    const IntBuffer &getActionCs(
        int i // Index of input layer to get predictions for
    ) const {
        return _actions[i];
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
