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
class Layer {
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
        SparseMatrix mappingWeights; // Weight matrix

        IntBuffer reconCs; // Reconstruction Cs
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCs; // Hidden states
    IntBuffer hiddenCsPrev; // Previous hidden states
    IntBuffer hiddenCsNext; // Predicted next hidden states

    FloatBuffer hiddenRewards;
    FloatBuffer hiddenValues;

    SparseMatrix transitionWeights;
    SparseMatrix feedBackWeights;

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forwardMapping(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learnMapping(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* inputCs,
        int vli
    );

    void backwardMapping(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int vli
    );

    void learnTransition(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* feedBackCs
    );

    void setReward(
        const Int2 &pos,
        std::mt19937 &rng,
        float reward
    );

    void valueIteration(
        const Int2 &pos,
        std::mt19937 &rng
    );

    void determinePolicy(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* feedBackCs
    );

    static void forwardMappingKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        l->forwardMapping(pos, rng, inputCs);
    }

    static void learnMappingKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l,
        const IntBuffer* inputCs,
        int vli
    ) {
        l->learnMapping(pos, rng, inputCs, vli);
    }

    static void backwardMappingKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l,
        const IntBuffer* hiddenCs,
        int vli
    ) {
        l->backwardMapping(pos, rng, hiddenCs, vli);
    }

    static void learnTransitionKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l,
        const IntBuffer* feedBackCs
    ) {
        l->learnTransition(pos, rng, feedBackCs);
    }

    static void setRewardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l,
        float reward
    ) {
        l->setReward(pos, rng, reward);
    }

    static void valueIterationKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l
    ) {
        l->valueIteration(pos, rng);
    }

    static void determinePolicyKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Layer* l,
        const IntBuffer* feedBackCs
    ) {
        l->determinePolicy(pos, rng, feedBackCs);
    }

public:
    float alpha; // Weight learning rate
    float beta; // Transition learning rate
    float gamma; // Discount factor
    int valueIters; // Number of value iterations

    // Defaults
    Layer()
    :
    alpha(0.1f),
    beta(0.1f),
    gamma(0.9f),
    valueIters(8)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        int lateralRadius,
        const std::vector<VisibleLayerDesc> &visibleLayerDescs, // Descriptors for visible layers
        bool hasFeedBack // Is this not the topmost layer?
    );

    // Activate the sparse coder (perform sparse coding)
    void stepForward(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    );

    void stepBackward(
        ComputeSystem &cs, // Compute system
        const IntBuffer* feedBackCs, // Feed back
        bool learnEnabled, // Whether to learn
        float reward // Reward for state
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

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }

    friend class Hierarchy;
};
} // namespace ogmaneo
