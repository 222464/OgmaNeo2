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
// A prediction layer (predicts x_(t+1))
class Predictor {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input

        int radius; // Radius onto input

        unsigned char canPropagate;

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2),
        canPropagate(false)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix weights; // Weight matrix

        IntBuffer inputCs;
        IntBuffer inputCsPrev; // Previous timestep (prev) input states
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    IntBuffer hiddenCs; // Hidden state

    FloatBuffer hiddenActivations;
    FloatBuffer hiddenActivationsPrev;

    FloatBuffer hiddenErrors;

    // Visible layers and descs
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng
    );

    void error(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenTargetCs
    );

    void prop(
        const Int2 &pos,
        std::mt19937 &rng,
        FloatBuffer* errors,
        int vli
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenTargetCs
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p
    ) {
        p->forward(pos, rng);
    }

    static void errorKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const IntBuffer* hiddenTargetCs
    ) {
        p->error(pos, rng, hiddenTargetCs);
    }

    static void propKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        FloatBuffer* errors,
        int vli
    ) {
        p->prop(pos, rng, errors, vli);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const IntBuffer* hiddenTargetCs
    ) {
        p->learn(pos, rng, hiddenTargetCs);
    }

public:
    float alpha; // Learning rate

    // Defaults
    Predictor()
    :
    alpha(0.1f)
    {}

    // Create with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // First visible layer must be from current hidden state, second must be feed back state, rest can be whatever
    ); 

    // Activate the predictor (predict values)
    void activate(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs // Hidden/output/prediction size
    );

    void propagate(
        ComputeSystem &cs, // Compute system
        const IntBuffer* hiddenTargetCs, // Targets
        FloatBuffer* errors, // Propagation accumulation
        int vli // Index of input to propagate to
    );

    // Learning predictions (update weights)
    void learn(
        ComputeSystem &cs,
        const IntBuffer* hiddenTargetCs
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

    // Get the hidden activations (predictions)
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of visible layer
    ) {
        return visibleLayers[i].weights;
    }
};
} // Namespace ogmaneo
