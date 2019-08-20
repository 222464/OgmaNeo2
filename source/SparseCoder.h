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
        _size(4, 4, 16),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        FloatBuffer _f1; // F1 field

        SparseMatrix _tBU; // Bottom up weight matrix
        SparseMatrix _tTD; // Top down weight matrix
        SparseMatrix _deltas; // Depletion parameter
        SparseMatrix _sigmas; // Depletion parameter

        IntBuffer _visibleCounts; // Number touching
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer

    IntBuffer _hiddenCounts; // Number touching
    
    IntBuffer _hiddenCs; // Hidden states
    IntBuffer _hiddenCsPrev; // Previous tick hidden states

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---

    void clear(
        const Int2 &pos,
        std::mt19937 &rng
    );
    
    void activate(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void match0(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs,
        int vli,
        bool isLast
    );

    void match1(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    static void clearKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc
    ) {
        sc->clear(pos, rng);
    }

    static void activateKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->activate(pos, rng, inputCs);
    }

    static void match0Kernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs,
        int vli,
        bool isLast
    ) {
        sc->match0(pos, rng, inputCs, vli, isLast);
    }

    static void match1Kernel(
        const Int2 &pos,
        std::mt19937 &rng,
        SparseCoder* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->match1(pos, rng, inputCs);
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
    float _alpha; // Activation function steepness
    float _beta; // Learning rate
    float _minVigilance; // For vigilance check
    int _iters; // Maximum number of times to reset

    // Defaults
    SparseCoder()
    :
    _alpha(0.1f),
    _beta(0.01f),
    _minVigilance(0.99f),
    _iters(10)
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

    // Get the hidden states
    const IntBuffer &getHiddenCsPrev() const {
        return _hiddenCsPrev;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // namespace ogmaneo
