// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace ogmaneo {
// Sparse coder
class SparseCoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input

        int radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix weights; // Weight matrix
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCs; // Hidden states
    IntBuffer hiddenCsPrev; // Previous hidden states

    IntBuffer hiddenCsRandom; // Random state for value iteration
    IntBuffer hiddenCsSelect; // Hidden states selected by value iteration

    SparseMatrix transitions; // Transition probabilities

    FloatBuffer hiddenValues; // Values learned through value iteration

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learnTransition(
        const Int2 &pos,
        std::mt19937 &rng
    );

    void learnFeedForward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* inputCs,
        int vli
    );

    void randomState(
        const Int2 &pos,
        std::mt19937 &rng
    ); 

    void valueIter(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* rewards
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->forward(pos, rng, inputCs);
    }

    static void learnTransitionKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc
    ) {
        sc->learnTransition(pos, rng);
    }

    static void learnFeedForwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const IntBuffer* inputCs,
        int vli
    ) {
        sc->learnFeedForward(pos, rng, inputCs, vli);
    }

    static void randomStateKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc
    ) {
        sc->randomState(pos, rng);
    }

    static void valueIterKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const FloatBuffer* rewards
    ) {
        sc->valueIter(pos, rng, rewards);
    }

public:
    float alpha; // Weight learning rate
    float beta; // Transition probability learning rate
    float gamma; // Discount factor
    int valueIters; // Number of value iterations

    // Defaults
    SparseCoder()
    :
    alpha(0.1f),
    beta(0.01f),
    gamma(0.9f),
    valueIters(16)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        int transitionRadius, // Radius of transition probability matrix receptive fields
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding), and also remember transitions
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    );

    // Optimize state 1 timestep ahead for path with most reward
    void optimize(
        ComputeSystem &cs,
        const FloatBuffer* rewards
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get the number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden states
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the previous hidden states
    const IntBuffer &getHiddenCsPrev() const {
        return hiddenCsPrev;
    }

    // Get selected hidden states
    const IntBuffer &getHiddenCsSelect() const {
        return hiddenCsSelect;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }

    friend class Hierarchy;
};
} // namespace ogmaneo
