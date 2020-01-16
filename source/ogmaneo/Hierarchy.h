// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEOLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"

#include <memory>

namespace ogmaneo {
// Type of hierarchy input layer
enum InputType {
    none = 0,
    action = 1
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius; // Feed forward radius
        int rRadius; // Routing radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)

        int temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        ffRadius(2),
        rRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}
    };

    struct RouteLayer {
        std::vector<SparseMatrix> weights;
        
        std::vector<IntBuffer> visibleCounts;

        std::vector<FloatBuffer> errors;
        std::vector<FloatBuffer> activations;

        IntBuffer hiddenCounts;
    };

    struct HistorySample {
        std::vector<IntBuffer> states;

        std::vector<IntBuffer> actionsPrev;

        float reward;
    };

private:
    // Layers
    std::vector<SparseCoder> scLayers;
    std::vector<RouteLayer> rLayers;

    std::vector<FloatBuffer> qs;
    std::vector<IntBuffer> actions;

    // Histories
    std::vector<std::vector<std::shared_ptr<IntBuffer>>> histories;
    std::vector<std::vector<int>> historySizes;

    // Per-layer values
    std::vector<char> updates;

    std::vector<int> ticks;
    std::vector<int> ticksPerUpdate;

    // Input dimensions
    std::vector<Int3> inputSizes;

    // History samples
    std::vector<std::shared_ptr<HistorySample>> historySamples;

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
    float alpha; // Output learning rate
    float beta; // Hidden learning rate
    float gamma; // Discount factor

    int maxHistorySamples; // Maximum number of history samples
    int historyIters; // Number of times to iterate over history

    // Default
    Hierarchy()
    :
    alpha(0.01f),
    beta(0.01f),
    gamma(0.99f),
    maxHistorySamples(64),
    historyIters(8)
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
        return scLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getActionCs(
        int i // Index of input layer to get predictions for
    ) const {
        return actions[i];
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
};
} // namespace ogmaneo
