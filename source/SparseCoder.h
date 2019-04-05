// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
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
        Int3 _size; // Size of input

        int _radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        _size({ 4, 4, 16 }),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix

        FloatBuffer _reconErrors; // Reconstruction errors

        IntBuffer _visibleCounts; // Number touching
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer

    IntBuffer _hiddenCs; // Hidden states

    FloatBuffer _hiddenBiases; // Biases

    IntBuffer _hiddenCounts; // Number touching

    IntBuffer _refractoryTimers; // Timers to track refractory period

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void recon(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs,
        int vli
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->forward(pos, rng, inputCs);
    }

    static void reconKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs,
        int vli
    ) {
        sc->recon(pos, rng, inputCs, vli);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->learn(pos, rng, inputCs);
    }

public:
    float _alpha; // Weight learning rate
    float _beta; // Bias learning rate
    int _refractoryTicks; // Time for refractory period

    // Defaults
    SparseCoder()
    :
    _alpha(0.01f),
    _beta(0.001f),
    _refractoryTicks(3)
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
        return _visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return _visibleLayerDescs[i];
    }

    // Get the hidden states
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i]._weights;
    }
};
} // namespace ogmaneo
