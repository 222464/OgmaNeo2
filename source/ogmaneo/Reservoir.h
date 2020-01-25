// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEOLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace ogmaneo {
// Sparse coder
class Reservoir {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input

        int radius; // Radius onto input

        float scale; // Scale of weights
        float dropRatio; // Ratio of weights to drop

        bool noDiagonal;

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2),
        scale(1.0f),
        dropRatio(0.0f),
        noDiagonal(false)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix weights; // Weight matrix
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    FloatBuffer hiddenStates; // Hidden states
    FloatBuffer hiddenStatesPrev; // Previous tick hidden states

    FloatBuffer hiddenBiases; // Bias weights

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputStates
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Reservoir* sc,
        const std::vector<const FloatBuffer*> &inputStates
    ) {
        sc->forward(pos, rng, inputStates);
    }

public:
    Reservoir() {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs, // Descriptors for visible layers
        float biasScale // Scale for bias weights
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputStates
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
    const FloatBuffer &getHiddenStates() const {
        return hiddenStates;
    }

    // Get the hidden states
    const FloatBuffer &getHiddenStatesPrev() const {
        return hiddenStatesPrev;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of visible layer
    ) const {
        return visibleLayers[i].weights;
    }
};
} // namespace ogmaneo
