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
        SparseMatrix ffWeights; // Tile
        SparseMatrix fbWeights; // Reconstruction

        IntBuffer inputCsPrev;

        FloatBuffer inputErrors;
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCs; // Hidden states
    IntBuffer hiddenCsPrev; // Previous hidden states

    FloatBuffer hiddenRefractories;

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void backward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* inputCs,
        int vli
    );

    void learnForward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learnBackward(
        const Int2 &pos,
        std::mt19937 &rng,
        int vli
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->forward(pos, rng, inputCs);
    }

    static void backwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const IntBuffer* inputCs,
        int vli
    ) {
        sc->backward(pos, rng, inputCs, vli);
    }

    static void learnForwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->learnForward(pos, rng, inputCs);
    }

    static void learnBackwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        int vli
    ) {
        sc->learnBackward(pos, rng, vli);
    }

public:
    float alpha; // FF weight learning rate
    float beta; // FB weight learning rate
    float gamma; // Refractory decay

    // Defaults
    SparseCoder()
    :
    alpha(0.1f),
    beta(0.1f),
    gamma(0.9f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
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

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace ogmaneo
