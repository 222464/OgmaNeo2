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
// A reinforcement learning layer (Q learning)
class Actor {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Visible/input size

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
        SparseMatrix weights;
        SparseMatrix traces;

        IntBuffer inputCsPrev;
        IntBuffer inputCsPrevPrev;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    IntBuffer hiddenCs; // Hidden states
    FloatBuffer hiddenValues; // Latest values
        
    // Visible layers and descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs,
        const IntBuffer* hiddenTargetCsPrev,
        float reward,
        bool learnEnabled
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Actor* a,
        const std::vector<const IntBuffer*> &inputCs,
        const IntBuffer* hiddenTargetCsPrev,
        float reward,
        bool learnEnabled
    ) {
        a->forward(pos, rng, inputCs, hiddenTargetCsPrev, reward, learnEnabled);
    }

public:
    float alpha; // Learning rate
    float gamma; // Discount factor (multiplicative)
    float traceDecay; // Trace decay rate (multiplicative)

    // Defaults
    Actor()
    :
    alpha(0.001f),
    gamma(0.99f),
    traceDecay(0.98f)
    {}

    // Initialized randomly
    void initRandom(
        ComputeSystem &cs,
        const Int3 &hiddenSize,
        const std::vector<VisibleLayerDesc> &visibleLayerDescs
    );

    // Step (get actions and update)
    void step(
        ComputeSystem &cs,
        const std::vector<const IntBuffer*> &inputCs,
        const IntBuffer* hiddenTargetCsPrev, // Action taken
        float reward,
        bool learnEnabled
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get hidden state/output/actions
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace ogmaneo
